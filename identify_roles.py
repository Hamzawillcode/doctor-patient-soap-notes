"""
Step 4 of the pipeline: Doctor / Patient role identification (rule-based).

    outputs/<name>_aligned.json  (speaker-labelled utterances)  ->  identify_roles.py  ->  DOCTOR / PATIENT labels

Approach: transparent rules on WHAT each speaker says (no model, nothing to install).
Every speaker gets points for "doctor-like" and "patient-like" language:

    doctor cues  : asking questions, clinical questions ("how long", "any allergies"),
                   examining / prescribing / planning ("let me check", "I'll prescribe")
    patient cues : describing own symptoms ("I have been", "my head"), own history
                   ("it started", "I took"), feelings ("I feel", "I'm worried")

    score = (doctor points - patient points) / number of utterances
            (divided by utterances so a speaker who simply talks more doesn't win automatically)

    The highest score -> DOCTOR. Of the rest, the one with the most patient points -> PATIENT.
    Any further speakers -> OTHER (e.g. a relative or nurse).

Confidence is based on how far apart the top two scores are. Low confidence is
flagged for a human to confirm - we never silently trust a weak decision.

Usage:
    python identify_roles.py audio/sample_consultation.wav
    python identify_roles.py --aligned outputs/sample_consultation_aligned.json
"""

import argparse
import json
import re
from pathlib import Path

# ---------------------------------------------------------------- cue lists
# Each entry: (name shown in the evidence, regular expression). Matching is case-insensitive.
DOCTOR_CUES = [
    ("opening question", r"\bwhat brings you\b|\bhow can i help\b|\bwhat seems to be\b|\bwhat can i do for you\b"),
    ("duration question", r"\bhow long\b|\bwhen did (it|this|they|the \w+) (start|begin)\b|\bsince when\b"),
    ("history question", r"\b(have|had|did) you (taken|tried|had|noticed|been|used|eaten|had any)\b"),
    ("symptom question", r"\bany (other )?(allergies|medication|medicines|fever|pain|symptoms|nausea|"
                         r"vomiting|history|problems)\b|\bdo you (have|feel|smoke|drink|take)\b|\bdoes it\b|"
                         r"\bis (it|the pain) (worse|better)\b|\bwhere (does it|is the pain)\b|"
                         r"\bon a scale\b|\bcan you describe\b|\btell me (more|about)\b"),
    ("examination", r"\blet me (examine|check|have a look|look|listen|take)\b|\b(take|check) your "
                    r"(temperature|blood pressure|pulse)\b|\bbreathe in\b|\bsay ah\b"),
    ("plan / prescription", r"\bi('ll| will| am going to|'m going to) (prescribe|order|refer|give you|send)\b|"
                            r"\bi('d| would) (like|recommend|suggest)\b|\bwe('ll| will) (do|run|start|check)\b|"
                            r"\b(blood test|x-ray|scan|follow[- ]up|come back)\b"),
    ("medical instruction", r"\b(take|use) (it|this|one|two|these)\b|\b(once|twice|three times) (a|per) day\b"
                            r"(?=.*\b(take|use)\b)|\bmake sure (you|to)\b|\byou should\b"),
    ("assessment", r"\b(it|this) (looks|sounds|seems) like\b|\bprobably (a|an|just)\b|\bdiagnos"),
]

PATIENT_CUES = [
    ("own symptoms", r"\bi('ve| have)( been| had)?( been)? (experiencing|having|feeling|suffering|getting|"
                     r"coughing|vomiting)\b|\bi('ve| have) (had|got) (a|an|some|this|fever|pain)\b"),
    ("body part 'my ...'", r"\bmy (head|stomach|chest|back|throat|leg|legs|arm|arms|knee|neck|ear|ears|eye|eyes|"
                           r"skin|tooth|teeth|nose|heart|foot|feet|hand|hands)\b"),
    ("feelings", r"\bi feel\b|\bi('m| am) (feeling|worried|scared|tired|exhausted|in pain)\b|\bit hurts\b|"
                 r"\bit('s| is) (painful|sore|itchy)\b"),
    ("own history", r"\bit (started|began)\b|\bi (took|tried|used)\b|\bi('ve| have) (taken|tried)\b|"
                    r"\bjust (paracetamol|ibuprofen|aspirin)\b|\bsince (yesterday|last)\b"),
    ("asking about own condition", r"\bis it (serious|bad|contagious|dangerous)\b|\bshould i (be|worry)\b|"
                                   r"\bwill i\b|\bdo i need\b|\bcan i\b"),
]

FIRST_SPEAKER_BONUS = 0.5   # doctors usually open the consultation - a weak hint only
CONFIDENCE_HIGH = 1.0       # score gap between top two speakers
CONFIDENCE_MEDIUM = 0.4


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:05.2f}"


def find_cues(text: str, cues: list[tuple[str, str]]) -> list[str]:
    """Return the names of all cues found in this text (each cue counted once per utterance)."""
    return [name for name, pattern in cues if re.search(pattern, text, flags=re.IGNORECASE)]


def score_speakers(utterances: list[dict]) -> dict:
    """Collect doctor / patient evidence per speaker and compute a score."""
    stats = {}
    first_speaker = next((u["speaker"] for u in utterances if u["speaker"] != "UNKNOWN"), None)

    for u in utterances:
        spk = u["speaker"]
        if spk == "UNKNOWN":
            continue
        s = stats.setdefault(spk, {"utterances": 0, "words": 0, "talk_time": 0.0,
                                   "doctor_points": 0.0, "patient_points": 0.0, "evidence": []})
        s["utterances"] += 1
        s["words"] += len(u["text"].split())
        s["talk_time"] += u["end"] - u["start"]

        questions = u["text"].count("?")
        if questions:
            s["doctor_points"] += questions
            s["evidence"].append({"role": "doctor", "cue": f"asks a question (x{questions})", "text": u["text"]})
        for cue in find_cues(u["text"], DOCTOR_CUES):
            s["doctor_points"] += 1
            s["evidence"].append({"role": "doctor", "cue": cue, "text": u["text"]})
        for cue in find_cues(u["text"], PATIENT_CUES):
            s["patient_points"] += 1
            s["evidence"].append({"role": "patient", "cue": cue, "text": u["text"]})

    if first_speaker in stats:
        stats[first_speaker]["doctor_points"] += FIRST_SPEAKER_BONUS
        stats[first_speaker]["evidence"].append({"role": "doctor", "cue": "speaks first (weak hint)", "text": ""})

    for s in stats.values():
        s["score"] = round((s["doctor_points"] - s["patient_points"]) / s["utterances"], 3)
        s["talk_time"] = round(s["talk_time"], 2)
    return stats


def assign_roles(stats: dict) -> tuple[dict, str, float]:
    """Highest score -> DOCTOR; most patient points among the rest -> PATIENT; others -> OTHER."""
    if not stats:
        return {}, "none", 0.0
    ranked = sorted(stats, key=lambda spk: stats[spk]["score"], reverse=True)
    roles = {ranked[0]: "DOCTOR"}
    rest = ranked[1:]
    if rest:
        patient = max(rest, key=lambda spk: (stats[spk]["patient_points"], -stats[spk]["score"]))
        roles[patient] = "PATIENT"
        for spk in rest:
            roles.setdefault(spk, "OTHER")

    if len(ranked) == 1:
        return roles, "low", 0.0          # only one voice: we cannot compare anything
    gap = stats[ranked[0]]["score"] - stats[ranked[1]]["score"]
    confidence = "high" if gap >= CONFIDENCE_HIGH else "medium" if gap >= CONFIDENCE_MEDIUM else "low"
    return roles, confidence, round(gap, 3)


def print_result(result: dict) -> None:
    print("=" * 60)
    print("EVIDENCE PER SPEAKER:\n")
    for spk, s in result["speakers"].items():
        print(f"{spk}  ->  {result['roles'][spk]}")
        print(f"   utterances {s['utterances']}, words {s['words']}, talk time {s['talk_time']} s")
        print(f"   doctor points {s['doctor_points']}  |  patient points {s['patient_points']}"
              f"  |  score {s['score']:+.2f}")
        for e in s["evidence"]:
            snippet = f'  "{e["text"]}"' if e["text"] else ""
            print(f"     [{e['role']:<7}] {e['cue']}{snippet}")
        print()

    print("=" * 60)
    print(f"ROLES: {result['roles']}")
    print(f"Confidence: {result['confidence'].upper()}  (score gap {result['score_gap']})")
    if result["needs_review"]:
        print("WARNING: low confidence - a human should confirm these roles before they are used.")

    print("\n" + "=" * 60)
    print("ROLE-LABELLED TRANSCRIPT\n")
    for u in result["utterances"]:
        print(f"[{format_time(u['start'])} → {format_time(u['end'])}]")
        print(f"{u['role']} ({u['speaker']}): {u['text']}\n")


def main():
    parser = argparse.ArgumentParser(description="Rule-based doctor/patient identification.")
    parser.add_argument("audio", nargs="?", help="Audio file (used to find outputs/<name>_aligned.json)")
    parser.add_argument("--aligned", help="Aligned JSON from align.py")
    parser.add_argument("--output", help="Where to save JSON (default: outputs/<name>_roles.json)")
    args = parser.parse_args()

    if not args.audio and not args.aligned:
        parser.error("give an audio file or --aligned")
    stem = Path(args.audio).stem if args.audio else Path(args.aligned).stem.removesuffix("_aligned")
    aligned_path = Path(args.aligned or f"outputs/{stem}_aligned.json")
    if not aligned_path.exists():
        raise SystemExit(f"Missing {aligned_path} - run `python align.py <audio>` first.")

    aligned = json.loads(aligned_path.read_text())
    stats = score_speakers(aligned["utterances"])
    roles, confidence, gap = assign_roles(stats)
    roles_with_unknown = {**roles, "UNKNOWN": "UNKNOWN"}

    result = {
        "audio_file": aligned.get("audio_file"),
        "source": str(aligned_path),
        "method": "rule-based linguistic cues",
        "roles": roles,
        "confidence": confidence,
        "score_gap": gap,
        "needs_review": confidence == "low",
        "speakers": stats,
        "utterances": [{**u, "role": roles_with_unknown.get(u["speaker"], "UNKNOWN")}
                       for u in aligned["utterances"]],
    }
    print_result(result)

    output_path = Path(args.output) if args.output else Path("outputs") / f"{stem}_roles.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved JSON to: {output_path}")


if __name__ == "__main__":
    main()
