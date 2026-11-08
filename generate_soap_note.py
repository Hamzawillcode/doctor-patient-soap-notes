"""
Steps 6b + 6c of the pipeline: write the SOAP note with a pretrained Transformer LM, then check it.

    outputs/<name>_soap_facts.json      (structured facts from build_soap_facts.py, each with evidence)
            ↓
    6b  Qwen/Qwen3-4B-Instruct-2507     pretrained + instruction-tuned decoder-only Transformer,
                                        used as-is (zero-shot prompt, greedy decoding, runs locally)
        "Write a SOAP note using ONLY these facts, cite a fact ID on every line"
            ↓
    6c  grounding check                 1. citations: every line cites existing facts of the right section
                                        2. clinical terms: our fine-tuned BioBERT re-reads the note; each
                                           entity must appear in the cited evidence
                                        3. numbers: every number must appear in the transcript evidence
                                        4. denials: lines citing a denied fact must say so
                                        + coverage: share of facts used in the note
            ↓
    outputs/<name>_soap_note.json  and  outputs/<name>_soap_note.md

The model only REPHRASES extracted facts. Anything it adds is flagged, never silently accepted.

Usage:
    python generate_soap_note.py audio/sample_consultation_2.wav
    python generate_soap_note.py --facts outputs/example_transcript_2_soap_facts.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LM_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SECTIONS = {"S": "SUBJECTIVE", "O": "OBJECTIVE", "A": "ASSESSMENT", "P": "PLAN"}

SYSTEM_PROMPT = """You are a clinical documentation assistant. You write SOAP notes from a list of facts \
that were extracted from a doctor-patient conversation.

Strict rules:
1. Use ONLY the facts provided. Do not add any symptom, diagnosis, medication, dose, number, test or \
advice that is not in the facts. Do not use outside medical knowledge.
2. End EVERY bullet line with the IDs of the facts it uses, in square brackets, e.g. [F2] or [F2, F5].
3. Put each fact only in its own section (the "section" field): S = SUBJECTIVE, O = OBJECTIVE, \
A = ASSESSMENT, P = PLAN.
4. Facts with status "absent" are things the patient DENIED: write them as denied (e.g. "Denies fever").
5. Facts with status "unclear" must be written as uncertain and should include the patient's answer.
6. ASSESSMENT and PLAN must only restate what the doctor said. Never suggest a diagnosis or treatment yourself.
7. If a section has no facts, write exactly: "- Not documented in the conversation."
8. Use concise clinical language. Output only the note, with the four headings SUBJECTIVE, OBJECTIVE, \
ASSESSMENT, PLAN, each followed by bullet lines starting with "- "."""


# ---------------------------------------------------------------- 6b: prompt + generation
def facts_for_prompt(facts: list[dict]) -> list[dict]:
    """Give each fact a short ID and keep only what the model needs (plus the quoted evidence)."""
    out = []
    for i, f in enumerate(facts, 1):
        f["id"] = f"F{i}"
        out.append({
            "id": f["id"], "section": f["section"], "type": f["type"], "fact": f["text"],
            "status": f["status"],
            **({"details": f["attributes"]} if f["attributes"] else {}),
            "evidence": [f'{ev["speaker"]}: "{ev.get("sentence", ev["text"])}"' for ev in f["evidence"]],
        })
    return out


def load_lm(device: str):
    print(f"Loading {LM_NAME} on {device} (first load reads ~8 GB of weights) ...")
    tok = AutoTokenizer.from_pretrained(LM_NAME)
    model = AutoModelForCausalLM.from_pretrained(LM_NAME, dtype=torch.bfloat16).to(device).eval()
    return tok, model


def section_plan(prompt_facts: list[dict]) -> str:
    """Tell the model exactly which fact IDs belong in each section - and which sections are empty.
    (Without this, the model filled empty Assessment/Plan sections with invented content.)"""
    lines = []
    for key, name in SECTIONS.items():
        ids = [f["id"] for f in prompt_facts if f["section"] == key]
        lines.append(f"{name}: use facts {', '.join(ids)}" if ids
                     else f'{name}: NO facts -> write exactly "- Not documented in the conversation."')
    return "\n".join(lines)


def generate_note(tok, model, prompt_facts: list[dict], device: str, max_new_tokens: int) -> tuple[str, dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Facts (JSON):\n" + json.dumps(prompt_facts, indent=1, ensure_ascii=False)
                                    + "\n\nSection plan (follow exactly):\n" + section_plan(prompt_facts)
                                    + "\n\nWrite the SOAP note now, with the four headings and bullet lines."},
    ]
    # The chat template wraps messages in the special tokens this model was instruction-tuned with.
    input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                        return_dict=True)["input_ids"].to(device)
    start = time.time()
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)  # greedy = reproducible
    new_tokens = output[0][input_ids.shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    stats = {"prompt_tokens": int(input_ids.shape[1]), "generated_tokens": int(len(new_tokens)),
             "seconds": round(time.time() - start, 1)}
    return text, stats


# ---------------------------------------------------------------- 6c: grounding check
CITATION = re.compile(r"\[(F\d+(?:\s*,\s*F\d+)*)\]")
DENIAL_WORDS = re.compile(r"\b(denies|denied|no|not|without|negative|absent)\b", re.IGNORECASE)
NUMBER_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twice", "once")
GENERIC_TERMS = {"symptom", "symptoms", "medication", "medications", "medicine", "problem", "problems", "condition"}
# Headings: "SUBJECTIVE", "Subjective:", "S:", "S -" ... optionally followed by content on the same line.
HEADING = re.compile(r"^(SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN|S|O|A|P)\s*(?::|-(?=\s)|$)\s*(.*)$", re.IGNORECASE)


def normalise(word: str) -> str:
    """Very small stemmer so word forms match: dizziness/dizzy -> dizzi, coughs -> cough."""
    w = word.lower()
    for suffix, repl in (("iness", "i"), ("ness", ""), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""), ("y", "i")):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: len(w) - len(suffix)] + repl
    return w


def supported(term: str, allowed: str) -> bool:
    """A term is supported if every word of it (after normalising) appears in the allowed evidence."""
    allowed_words = {normalise(w) for w in re.findall(r"[a-z0-9]+", allowed.lower())}
    return all(normalise(w) in allowed_words for w in re.findall(r"[a-z0-9]+", term.lower()))


def parse_note(note: str) -> tuple[list[dict], set]:
    """Split the note into lines under each heading. Returns (lines, headings found)."""
    lines, section, found = [], None, set()
    for raw in note.splitlines():
        line = raw.strip().strip("*#").strip()
        if not line:
            continue
        m = HEADING.match(line)
        if m:
            section = m.group(1)[0].upper()
            found.add(section)
            line = m.group(2).strip()
            if not line:
                continue
        if section is None:
            continue
        text = line.lstrip("-• ").strip()
        cited = [c.strip() for mm in CITATION.findall(text) for c in mm.split(",")]
        lines.append({"section": section, "text": text, "cited": cited, "bullet": line.startswith(("-", "•"))})
    return lines, found


def grounding_check(note: str, facts: list[dict], ner) -> dict:
    by_id = {f["id"]: f for f in facts}
    lines, found = parse_note(note)
    problems, cited_ids = [], set()
    # 0. format: all four sections present and readable, otherwise nothing below can be trusted
    missing_sections = [SECTIONS[k] for k in SECTIONS if k not in found]
    if missing_sections:
        problems.append({"check": "format", "line": "", "issue": f"missing section headings: {missing_sections}"})
    if not lines:
        problems.append({"check": "format", "line": "", "issue": "no note lines could be read"})
    for ln in lines:
        if not ln["bullet"]:
            problems.append({"check": "format", "line": ln["text"], "issue": "not a bullet line"})

    for ln in lines:
        body = CITATION.sub("", ln["text"]).strip()
        if body.lower().startswith("not documented"):
            continue
        # 1. citations exist and belong to the same section
        if not ln["cited"]:
            problems.append({"check": "citation", "line": ln["text"], "issue": "no fact cited"})
            continue
        missing = [c for c in ln["cited"] if c not in by_id]
        if missing:
            problems.append({"check": "citation", "line": ln["text"], "issue": f"cites unknown facts {missing}"})
        cited = [by_id[c] for c in ln["cited"] if c in by_id]
        cited_ids.update(f["id"] for f in cited)
        wrong = [f["id"] for f in cited if f["section"] != ln["section"]]
        if wrong:
            problems.append({"check": "section", "line": ln["text"],
                             "issue": f"facts {wrong} belong to another section"})

        # the text the line is allowed to use: cited facts, their details and their evidence sentences
        allowed = " ".join([f["text"] for f in cited]
                           + [x for f in cited for v in f["attributes"].values() for x in (v if isinstance(v, list) else [v])]
                           + [ev.get("sentence", ev["text"]) for f in cited for ev in f["evidence"]]).lower()

        # 2. clinical terms found by BioBERT in the generated line must be supported
        entities, _ = ner.extract(body)
        for e in entities:
            if e["text"].lower() in GENERIC_TERMS:
                continue
            if not supported(e["text"], allowed):
                problems.append({"check": "clinical term", "line": ln["text"],
                                 "issue": f"'{e['text']}' ({e['label']}) not found in cited evidence"})

        # 3. numbers must be supported
        for num in re.findall(r"\d+(?:\.\d+)?", body):
            if num not in allowed:
                problems.append({"check": "number", "line": ln["text"], "issue": f"number '{num}' not in evidence"})
        for w in NUMBER_WORDS:
            if re.search(rf"\b{w}\b", body, re.IGNORECASE) and w not in allowed:
                problems.append({"check": "number", "line": ln["text"], "issue": f"'{w}' not in evidence"})

        # 4. denied facts must be written as denied
        if any(f["status"] == "absent" for f in cited) and not DENIAL_WORDS.search(body):
            problems.append({"check": "denial", "line": ln["text"], "issue": "cites a denied fact but doesn't say so"})

    content_facts = [f for f in facts if f["type"] != "patient_concern"]
    covered = [f["id"] for f in content_facts if f["id"] in cited_ids]
    return {
        "lines": len(lines),
        "problems": problems,
        "passed": not problems,
        "coverage": round(len(covered) / len(content_facts), 3) if content_facts else 1.0,
        "facts_not_used": [f"{f['id']}: {f['text']}" for f in content_facts if f["id"] not in cited_ids],
    }


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Write a SOAP note with a pretrained LM and check its grounding.")
    parser.add_argument("audio", nargs="?", help="Audio file (finds outputs/<name>_soap_facts.json)")
    parser.add_argument("--facts", help="Facts JSON from build_soap_facts.py")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    args = parser.parse_args()
    if not (args.audio or args.facts):
        parser.error("give an audio file or --facts")
    stem = Path(args.audio).stem if args.audio else Path(args.facts).stem.removesuffix("_soap_facts")
    path = Path(args.facts or f"outputs/{stem}_soap_facts.json")
    if not path.exists():
        raise SystemExit(f"Missing {path} - run `python build_soap_facts.py ...` first.")

    data = json.loads(path.read_text())
    facts = data["facts"]
    prompt_facts = facts_for_prompt(facts)

    tok, model = load_lm(args.device)
    note, stats = generate_note(tok, model, prompt_facts, args.device, args.max_new_tokens)
    del model
    if args.device == "mps":
        torch.mps.empty_cache()

    print("=" * 60)
    print(f"SOAP NOTE  (written by {LM_NAME}; {stats['generated_tokens']} tokens in {stats['seconds']} s)\n")
    print(note)

    from clinical_extraction import FINETUNED_PATH, BioBertNER
    ner = BioBertNER(FINETUNED_PATH)
    check = grounding_check(note, facts, ner)

    print("\n" + "=" * 60)
    print("GROUNDING CHECK\n")
    print(f"Lines checked: {check['lines']}   |   fact coverage: {check['coverage']:.0%}")
    if check["passed"]:
        print("PASSED: every line cites existing facts; no unsupported clinical terms, numbers or denials.")
    for p in check["problems"]:
        print(f"FLAG [{p['check']}] {p['issue']}\n      line: {p['line']}")
    if check["facts_not_used"]:
        print("Facts not used in the note: " + "; ".join(check["facts_not_used"]))

    result = {"source": str(path), "language_model": LM_NAME, "generation": {**stats, "decoding": "greedy"},
              "system_prompt": SYSTEM_PROMPT, "facts_given": prompt_facts, "note": note,
              "template_note_baseline": data.get("template_note"), "grounding_check": check}
    out = Path("outputs") / f"{stem}_soap_note.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    # In the saved note, every flagged line is visibly marked so it can't be mistaken for verified content.
    flagged = {p["line"] for p in check["problems"] if p["line"]}
    marked = "\n".join(raw + ("   ⚠ UNVERIFIED" if any(fl and fl in raw for fl in flagged) else "")
                       for raw in note.splitlines())
    flags_md = "\n".join(f"- [{p['check']}] {p['issue']}" + (f" — `{p['line']}`" if p["line"] else "")
                         for p in check["problems"]) or "- none"
    md = Path("outputs") / f"{stem}_soap_note.md"
    md.write_text(f"# SOAP note\n\n_Generated by {LM_NAME} from extracted facts; "
                  f"grounding check {'passed' if check['passed'] else 'flagged ' + str(len(check['problems'])) + ' issue(s)'}; "
                  f"fact coverage {check['coverage']:.0%}. Requires clinician review._\n\n{marked}\n\n"
                  f"## Grounding flags\n\n{flags_md}\n")
    print(f"\nSaved: {out}  and  {md}")


if __name__ == "__main__":
    main()
