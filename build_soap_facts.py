"""
Step 6a of the pipeline: build STRUCTURED SOAP FACTS from the conversation (no model, no generation).

    outputs/<name>_clinical_entities.json   (DOCTOR/PATIENT utterances + BioBERT entities)
            ↓
    1. question–answer pairing   DOCTOR "Any nausea?" + PATIENT "Yes."  -> nausea PRESENT for the patient
    2. negation                  PATIENT "No fever."                     -> fever ABSENT
    3. attribute linking         Duration/Severity -> nearest symptom, Dosage/Frequency -> nearest medication
    4. section assignment        S / O / A / P using speaker + entity label + cue words
            ↓
    outputs/<name>_soap_facts.json   (every fact keeps its source sentence = traceable)
    + a plain TEMPLATE note (baseline to compare with the language-model note in step 6b)

Safety rules:
    - Assessment and Plan contain ONLY what the DOCTOR said (quoted). Nothing is diagnosed or prescribed here.
    - Nothing is invented: if a value was not said (e.g. a blood-pressure number), it is not written.

Usage:
    python build_soap_facts.py audio/sample_consultation_2.wav
    python build_soap_facts.py --entities outputs/example_transcript_clinical_entities.json
"""

import argparse
import json
import re
from pathlib import Path

# ---------------------------------------------------------------- vocabulary
CLINICAL_LABELS = {"Sign_symptom", "Disease_disorder"}          # "conditions" the patient may have
GENERIC_WORDS = {"medication", "medications", "medicine", "medicines", "drug", "drugs", "pills", "tablets",
                 "symptoms", "symptom", "problem", "problems"}   # entity text too vague to be a fact

YES = r"^(yes|yeah|yep|yup|i do|i have|i did|i am|i'm|a little|a bit|sometimes|definitely|correct|right|mm-?hmm)\b"
NO = r"^(no|nope|nah|not really|never|i don't|i do not|i haven't|i have not|i didn't|i'm not|none)\b"

NEGATION_CUES = r"\b(no|not|never|without|denies|deny|denied|don't|do not|haven't|have not|hasn't|didn't|isn't|no history of)\b"
CLAUSE_BREAK = r"[,.;!?]|\bbut\b|\bhowever\b|\balthough\b"
NEGATION_WINDOW = 40          # characters between the cue and the entity

OBJECTIVE_CUES = (r"\blet me (examine|check|look|listen|feel|take)\b|\b(blood pressure|temperature|pulse|heart rate|"
                  r"oxygen|saturation|weight|lungs?|chest sounds?|heart sounds?|abdomen|throat|ears?)\b.*\b(is|are|was|"
                  r"were|sounds?|looks?|seems?|normal|clear|high|low|elevated|red|swollen|tender)\b|\b\d+(\.\d+)?\s?"
                  r"(mmhg|degrees|°|bpm|%|kg)\b|\bon examination\b|\bexam(ination)? (shows|showed|reveals)\b")
ASSESSMENT_CUES = (r"\b(it|this|that) (looks|sounds|seems) like\b|\bi think (it|this|you)\b|\bprobably\b|\blikely\b|"
                   r"\bmost likely\b|\bdiagnos|\byou (have|'ve got) (a|an)\b|\bconsistent with\b|\bsuggests?\b")
PLAN_CUES = (r"\b(i('ll| will)|we('ll| will)|i('m| am) going to|let's)\b.*\b(prescribe|give|start|order|refer|send|"
             r"arrange|book|run|do)\b|\bprescri|\b(take|use|apply)\b.*\b(daily|a day|times|every|hours|tablets?|"
             r"capsules?)\b|\bcome back\b|\bfollow[- ]?up\b|\breturn\b|\b(blood test|x-ray|scan|mri|ultrasound|"
             r"referral|rest|fluids|avoid)\b")


def find(pattern: str, text: str):
    return re.search(pattern, text, flags=re.IGNORECASE)


# ---------------------------------------------------------------- helpers
def evidence(turn: dict, index: int, sentence: str | None = None) -> dict:
    ev = {"utterance": index, "speaker": turn["speaker"], "text": turn["text"],
          "start": turn.get("start"), "end": turn.get("end")}
    if sentence and sentence != turn["text"]:
        ev["sentence"] = sentence
    return ev


def sentences(turn: dict) -> list[tuple[str, list[dict]]]:
    """Split an utterance into sentences; give each sentence the entities that fall inside it."""
    out = []
    # A sentence ends at . ! ? followed by a space or the end - so "37.2 degrees" is not split.
    for m in re.finditer(r".+?(?:[.!?]+(?=\s|$)|$)", turn["text"]):
        if m.group().strip():
            ents = [e for e in turn.get("entities", []) if m.start() <= e["start"] < m.end()]
            out.append((m.group().strip(), ents))
    return out


def is_question(text: str) -> bool:
    return "?" in text


def answer_polarity(text: str) -> str:
    t = text.strip().lower()
    if find(NO, t):
        return "absent"
    if find(YES, t):
        return "present"
    return "unclear"


def is_negated(text: str, entity: dict) -> bool:
    """NegEx-style: a negation cue shortly BEFORE the entity, with no clause break in between."""
    before = text[max(0, entity["start"] - NEGATION_WINDOW):entity["start"]]
    cues = list(re.finditer(NEGATION_CUES, before, flags=re.IGNORECASE))
    if not cues:
        return False
    between = before[cues[-1].end():]
    return not re.search(CLAUSE_BREAK, between, flags=re.IGNORECASE)


def nearest(entity: dict, candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    centre = (entity["start"] + entity["end"]) / 2
    return min(candidates, key=lambda c: abs((c["start"] + c["end"]) / 2 - centre))


def attach_attributes(entities: list[dict]) -> dict:
    """Return {id(main entity): {"duration": [...], "severity": [...], ...}} for one utterance."""
    attrs = {}
    conditions = [e for e in entities if e["label"] in CLINICAL_LABELS]
    meds = [e for e in entities if e["label"] == "Medication" and e["text"].lower() not in GENERIC_WORDS]
    for e in entities:
        target = None
        if e["label"] in ("Duration", "Severity"):
            target = nearest(e, conditions)
        elif e["label"] in ("Dosage", "Frequency"):
            target = nearest(e, meds)
        if target is not None:
            attrs.setdefault(id(target), {}).setdefault(e["label"].lower(), []).append(e["text"])
            e["_attached"] = True
    return attrs


# ---------------------------------------------------------------- main logic
def build_facts(conversation: list[dict]) -> tuple[list[dict], list[dict]]:
    facts, unused = [], []
    answered = set()       # patient answers already consumed by a question

    def add(section, ftype, text, status, ev, method, attributes=None, label=None):
        facts.append({"section": section, "type": ftype, "text": text, "status": status, "label": label,
                      "attributes": attributes or {}, "evidence": ev, "method": method})

    for i, turn in enumerate(conversation):
        role, text, ents = turn["speaker"], turn["text"], turn.get("entities", [])
        attrs = attach_attributes(ents)

        # ---------------- DOCTOR
        if role == "DOCTOR":
            used = False
            # The doctor often says several things in one turn ("It looks like X. I'll prescribe Y."),
            # so each SENTENCE is classified on its own.
            for sent, s_ents in sentences(turn):
                ev = evidence(turn, i, sent)
                if is_question(sent):
                    asked = [e for e in s_ents if e["label"] in CLINICAL_LABELS | {"Medication"}
                             and e["text"].lower() not in GENERIC_WORDS]
                    nxt = conversation[i + 1] if i + 1 < len(conversation) else None
                    if asked and nxt and nxt["speaker"] != "DOCTOR":
                        status = answer_polarity(nxt["text"])
                        for e in asked:
                            section_type = "medication" if e["label"] == "Medication" else "symptom"
                            add("S", section_type, e["text"], status, [ev, evidence(nxt, i + 1)],
                                f"question-answer ({status})", attrs.get(id(e)), e["label"])
                        answered.add(i + 1)     # the answer was used (its own entities are still processed)
                        used = True
                    continue
                if find(PLAN_CUES, sent):
                    meds = [e["text"] for e in s_ents
                            if e["label"] == "Medication" and e["text"].lower() not in GENERIC_WORDS]
                    add("P", "doctor_plan", sent, "stated by doctor", [ev], "plan cue",
                        {"medications": meds} if meds else None)
                    used = True
                elif find(OBJECTIVE_CUES, sent):
                    add("O", "examination", sent, "stated by doctor", [ev], "examination cue")
                    used = True
                elif find(ASSESSMENT_CUES, sent) or any(e["label"] == "Disease_disorder" for e in s_ents):
                    diseases = [e["text"] for e in s_ents if e["label"] == "Disease_disorder"]
                    add("A", "doctor_impression", ", ".join(diseases) or sent, "stated by doctor", [ev],
                        "assessment cue / doctor-stated disease", label="Disease_disorder" if diseases else None)
                    used = True
            if not used:
                unused.append(evidence(turn, i))
            continue

        # ---------------- PATIENT (or OTHER, e.g. a relative)
        reporter = "patient" if role == "PATIENT" else role.lower()
        used = i in answered
        question_spans = [(m.start(), m.end()) for m in re.finditer(r".+?(?:[.!?]+(?=\s|$)|$)", text)
                          if "?" in m.group()]
        for e in ents:
            if e.get("_attached") or e["text"].lower() in GENERIC_WORDS:
                continue
            # Entities inside the patient's own QUESTION ("Do I have a fever?") are not reports;
            # the question is recorded as a concern below.
            if any(s <= e["start"] < t for s, t in question_spans):
                continue
            status = "absent" if is_negated(text, e) else "present"
            if e["label"] in CLINICAL_LABELS:
                add("S", "symptom" if e["label"] == "Sign_symptom" else "condition", e["text"], status,
                    [evidence(turn, i)], f"{reporter} statement" + (" (negated)" if status == "absent" else ""),
                    attrs.get(id(e)), e["label"])
            elif e["label"] == "Medication":
                add("S", "medication", e["text"], status, [evidence(turn, i)], f"{reporter} statement",
                    attrs.get(id(e)), e["label"])
            else:   # Duration / Severity / Dosage / Frequency with nothing to attach to
                add("S", "unattached_" + e["label"].lower(), e["text"], status, [evidence(turn, i)],
                    f"{reporter} statement (no symptom/medication found in the same sentence)", label=e["label"])
            used = True
        for sent, _ in sentences(turn):
            if is_question(sent):
                add("S", "patient_concern", sent, "asked", [evidence(turn, i, sent)], f"{reporter} question")
                used = True
        if not used:
            unused.append(evidence(turn, i))

    for e in (e for t in conversation for e in t.get("entities", [])):
        e.pop("_attached", None)
    return merge_duplicates(facts), unused


def merge_duplicates(facts: list[dict]) -> list[dict]:
    """Same section + type + text + status = one fact with all its evidence
    (e.g. 'Any fever?' / 'No, I haven't had any fever.' gives fever twice)."""
    merged = {}
    for f in facts:
        key = (f["section"], f["type"], f["text"].lower(), f["status"])
        if key not in merged:
            merged[key] = f
            continue
        m = merged[key]
        for ev in f["evidence"]:
            if ev not in m["evidence"]:
                m["evidence"].append(ev)
        if f["method"] not in m["method"]:
            m["method"] += " + " + f["method"]
        for k, v in f["attributes"].items():
            m["attributes"].setdefault(k, [])
            m["attributes"][k] += [x for x in v if x not in m["attributes"][k]]
    return list(merged.values())


# ---------------------------------------------------------------- template note (baseline)
def describe(f: dict) -> str:
    extra = []
    for key in ("duration", "severity", "dosage", "frequency"):
        if f["attributes"].get(key):
            extra.append(f"{key}: {', '.join(f['attributes'][key])}")
    return f"{f['text']}" + (f" ({'; '.join(extra)})" if extra else "")


def template_note(facts: list[dict]) -> str:
    S = [f for f in facts if f["section"] == "S"]
    lines = ["SUBJECTIVE"]
    groups = [
        ("Reported symptoms/conditions", [f for f in S if f["type"] in ("symptom", "condition") and f["status"] == "present"]),
        ("Denied", [f for f in S if f["status"] == "absent"]),
        ("Asked, answer unclear", [f for f in S if f["status"] == "unclear"]),
        ("Current medications", [f for f in S if f["type"] == "medication" and f["status"] == "present"]),
        ("Other details (not linked to a symptom/medication)", [f for f in S if f["type"].startswith("unattached_")]),
    ]
    for title, items in groups:
        if items:
            lines.append(f"  {title}: " + "; ".join(describe(f) for f in items))
    concerns = [f["text"] for f in S if f["type"] == "patient_concern"]
    if concerns:
        lines.append("  Patient questions/concerns: " + " ".join(f'"{c}"' for c in concerns))
    for sec, title in (("O", "OBJECTIVE"), ("A", "ASSESSMENT (as stated by the doctor)"), ("P", "PLAN (as stated by the doctor)")):
        lines.append(title)
        items = [f for f in facts if f["section"] == sec]
        if not items:
            lines.append("  Not documented in the conversation.")
        for f in items:
            quote = f["evidence"][0].get("sentence", f["evidence"][0]["text"])
            lines.append(f"  - {f['text']}  [\"{quote}\"]" if sec == "A" else f"  - \"{f['text']}\"")
    return "\n".join(lines)


def print_facts(facts: list[dict], unused: list[dict]) -> None:
    print("=" * 60)
    print("STRUCTURED SOAP FACTS (each with its evidence)\n")
    for sec in "SOAP":
        print(f"[{sec}]")
        for f in (f for f in facts if f["section"] == sec):
            label = f" <{f['label']}>" if f["label"] else ""
            print(f"  {describe(f)}{label}  status={f['status']}  rule={f['method']}")
            for ev in f["evidence"]:
                print(f"      ← {ev['speaker']}: \"{ev.get('sentence', ev['text'])}\"")
        print()
    if unused:
        print("Utterances not used for any fact:")
        for ev in unused:
            print(f"      {ev['speaker']}: \"{ev['text']}\"")


def main():
    parser = argparse.ArgumentParser(description="Build structured SOAP facts from clinical entities.")
    parser.add_argument("audio", nargs="?", help="Audio file (finds outputs/<name>_clinical_entities.json)")
    parser.add_argument("--entities", help="Clinical entities JSON from clinical_extraction.py")
    parser.add_argument("--output", help="Where to save JSON (default: outputs/<name>_soap_facts.json)")
    args = parser.parse_args()
    if not (args.audio or args.entities):
        parser.error("give an audio file or --entities")
    stem = Path(args.audio).stem if args.audio else Path(args.entities).stem.removesuffix("_clinical_entities")
    path = Path(args.entities or f"outputs/{stem}_clinical_entities.json")
    if not path.exists():
        raise SystemExit(f"Missing {path} - run `python clinical_extraction.py ...` first.")

    data = json.loads(path.read_text())
    facts, unused = build_facts(data["conversation"])
    note = template_note(facts)

    print_facts(facts, unused)
    print("=" * 60)
    print("TEMPLATE SOAP NOTE (rule-based baseline, no language model)\n")
    print(note)

    result = {
        "source": str(path),
        "ner_model_set": data.get("model_set"),
        "facts": facts,
        "unused_utterances": unused,
        "template_note": note,
        "rules": {"answer_yes": YES, "answer_no": NO, "negation_cues": NEGATION_CUES,
                  "objective_cues": OBJECTIVE_CUES, "assessment_cues": ASSESSMENT_CUES, "plan_cues": PLAN_CUES},
    }
    out = Path(args.output) if args.output else Path("outputs") / f"{stem}_soap_facts.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved JSON to: {out}")


if __name__ == "__main__":
    main()
