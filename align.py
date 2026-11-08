"""
Step 3 of the pipeline: Timestamp Alignment ("who said which words?").

    outputs/<name>.json              (Whisper words + timestamps)    ─┐
                                                                     ├─> align.py -> speaker-labelled transcript
    outputs/<name>_diarization.json  (pyannote speaker intervals)    ─┘

This script does NOT run any model. It only combines the two JSON files that
transcribe.py and diarize.py already produced.

SPEAKER_00 / SPEAKER_01 stay anonymous - they do NOT mean doctor or patient.

Assignment rules (applied to every word, in this order):
  1. MAX OVERLAP   - the word goes to the speaker whose interval overlaps it the most.
  2. TIE           - if two speakers have the same (largest) overlap, the speaker whose
                     interval contains the word's MIDPOINT wins; if that doesn't decide,
                     the speaker of the PREVIOUS word wins (speakers rarely switch mid-word).
  3. GAP           - if the word overlaps no interval, use the NEAREST interval, but only
                     if it is within --max-gap seconds (default 0.5). Flagged "nearest".
  4. NO MATCH      - otherwise the word is labelled UNKNOWN and reported. Never guessed.

Usage:
    python align.py audio/sample_consultation.wav
    python align.py --stt outputs/a.json --diarization outputs/a_diarization.json
"""

import argparse
import json
from pathlib import Path

UNKNOWN = "UNKNOWN"
TIE_TOLERANCE = 1e-6  # overlaps closer than this (in seconds) count as equal


def format_time(seconds: float) -> str:
    """Turn 63.5 into '01:03.50' (minutes:seconds.hundredths)."""
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length (seconds) of the time shared by two intervals; 0 if they don't touch."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def distance(word: dict, interval: dict) -> float:
    """Gap (seconds) between a word and an interval; 0 if they overlap."""
    if word["end"] < interval["start"]:
        return interval["start"] - word["end"]
    if word["start"] > interval["end"]:
        return word["start"] - interval["end"]
    return 0.0


def assign_speaker(word: dict, intervals: list[dict], previous_speaker: str | None,
                   max_gap: float) -> tuple[str, str, float]:
    """
    Decide which speaker said this word.
    Returns (speaker, method, overlap_seconds).
    method is one of: "overlap", "tie-midpoint", "tie-previous", "tie-first", "nearest", "none".
    """
    # Total overlap per speaker. (With exclusive diarization a speaker can have
    # several intervals; a word touching two intervals of the same speaker counts both.)
    per_speaker = {}
    for iv in intervals:
        ov = overlap(word["start"], word["end"], iv["start"], iv["end"])
        if ov > 0:
            per_speaker[iv["speaker"]] = per_speaker.get(iv["speaker"], 0.0) + ov

    # Rule 1: maximum overlap.
    if per_speaker:
        best = max(per_speaker.values())
        leaders = [s for s, ov in per_speaker.items() if best - ov <= TIE_TOLERANCE]
        if len(leaders) == 1:
            return leaders[0], "overlap", best

        # Rule 2: tie -> midpoint, then previous word's speaker, then first in time.
        mid = (word["start"] + word["end"]) / 2
        at_mid = [iv["speaker"] for iv in intervals
                  if iv["speaker"] in leaders and iv["start"] <= mid < iv["end"]]
        if len(set(at_mid)) == 1:
            return at_mid[0], "tie-midpoint", best
        if previous_speaker in leaders:
            return previous_speaker, "tie-previous", best
        first = min((iv for iv in intervals if iv["speaker"] in leaders), key=lambda iv: iv["start"])
        return first["speaker"], "tie-first", best

    # Rule 3: no overlap -> nearest interval within max_gap.
    nearest = min(intervals, key=lambda iv: distance(word, iv), default=None)
    if nearest is not None and distance(word, nearest) <= max_gap:
        return nearest["speaker"], "nearest", 0.0

    # Rule 4: give up honestly.
    return UNKNOWN, "none", 0.0


def align_words(words: list[dict], intervals: list[dict], max_gap: float) -> list[dict]:
    aligned = []
    previous = None
    for w in words:
        speaker, method, ov = assign_speaker(w, intervals, previous, max_gap)
        aligned.append({
            "word": w["word"],
            "start": w["start"],
            "end": w["end"],
            "speaker": speaker,
            "assignment": method,           # how the speaker was decided
            "overlap": round(ov, 3),        # seconds of overlap with the chosen speaker
        })
        if speaker != UNKNOWN:
            previous = speaker
    return aligned


def group_utterances(aligned_words: list[dict]) -> list[dict]:
    """Join consecutive words of the same speaker into one utterance."""
    utterances = []
    for w in aligned_words:
        if utterances and utterances[-1]["speaker"] == w["speaker"]:
            u = utterances[-1]
            u["end"] = w["end"]
            u["words"].append(w["word"])
        else:
            utterances.append({"speaker": w["speaker"], "start": w["start"],
                               "end": w["end"], "words": [w["word"]]})
    # Whisper words already carry their punctuation ("today?"), so a space join is enough.
    return [{"speaker": u["speaker"], "start": u["start"], "end": u["end"],
             "text": " ".join(u["words"])} for u in utterances]


def print_result(result: dict) -> None:
    print("=" * 60)
    print("WORD → SPEAKER ASSIGNMENT:\n")
    for w in result["words"]:
        flag = "" if w["assignment"] == "overlap" else f"   <-- {w['assignment']}"
        print(f"{w['word']:<15} {format_time(w['start'])} – {format_time(w['end'])}"
              f"   {w['speaker']:<11} overlap {w['overlap']:.2f}s{flag}")

    print("\n" + "=" * 60)
    print("SPEAKER-LABELLED TRANSCRIPT\n")
    for u in result["utterances"]:
        print(f"[{format_time(u['start'])} → {format_time(u['end'])}]")
        print(f"{u['speaker']}: {u['text']}\n")

    s = result["summary"]
    print("=" * 60)
    print(f"Words: {s['total_words']}  |  by method: {s['by_method']}")
    if s["unknown_words"]:
        print(f"WARNING: {s['unknown_words']} word(s) could not be assigned (UNKNOWN).")
    if s["by_method"].get("nearest"):
        print(f"NOTE: {s['by_method']['nearest']} word(s) fell in a gap and used the nearest speaker.")


def main():
    parser = argparse.ArgumentParser(description="Align Whisper words with pyannote speakers.")
    parser.add_argument("audio", nargs="?", help="Audio file used for both stages (to find the JSON files)")
    parser.add_argument("--stt", help="STT JSON (default: outputs/<audio name>.json)")
    parser.add_argument("--diarization", help="Diarization JSON (default: outputs/<audio name>_diarization.json)")
    parser.add_argument("--max-gap", type=float, default=0.5,
                        help="Max distance (s) to the nearest speaker for words in a gap (default 0.5)")
    parser.add_argument("--output", help="Where to save JSON (default: outputs/<audio name>_aligned.json)")
    args = parser.parse_args()

    if not args.audio and not (args.stt and args.diarization):
        parser.error("give an audio file, or both --stt and --diarization")
    stem = Path(args.audio).stem if args.audio else Path(args.stt).stem
    stt_path = Path(args.stt or f"outputs/{stem}.json")
    diar_path = Path(args.diarization or f"outputs/{stem}_diarization.json")
    for p, script in [(stt_path, "transcribe.py"), (diar_path, "diarize.py")]:
        if not p.exists():
            raise SystemExit(f"Missing {p} - run `python {script} <audio>` first.")

    stt = json.loads(stt_path.read_text())
    diar = json.loads(diar_path.read_text())
    if not stt.get("words"):
        raise SystemExit(f"{stt_path} has no word timestamps.")

    # exclusive_speakers = at most one speaker at any moment -> clean, non-overlapping transcript.
    intervals = sorted(diar["exclusive_speakers"], key=lambda iv: iv["start"])

    aligned = align_words(stt["words"], intervals, args.max_gap)
    utterances = group_utterances(aligned)

    by_method = {}
    for w in aligned:
        by_method[w["assignment"]] = by_method.get(w["assignment"], 0) + 1
    result = {
        "audio_file": stt.get("audio_file"),
        "sources": {"stt": str(stt_path), "diarization": str(diar_path),
                    "speaker_timeline": "exclusive_speakers"},
        "settings": {"max_gap": args.max_gap},
        "summary": {"total_words": len(aligned), "by_method": by_method,
                    "unknown_words": by_method.get("none", 0)},
        "full_transcript": stt["full_transcript"],
        "utterances": utterances,
        "words": aligned,
    }
    print_result(result)

    output_path = Path(args.output) if args.output else Path("outputs") / f"{stem}_aligned.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved JSON to: {output_path}")


if __name__ == "__main__":
    main()
