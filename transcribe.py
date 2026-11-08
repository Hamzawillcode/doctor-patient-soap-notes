"""
Step 1 of the pipeline: Speech-to-Text with Whisper.

    audio file  ->  Whisper  ->  transcript + segment timestamps + word timestamps

Usage:
    python transcribe.py path/to/audio.wav
    python transcribe.py path/to/audio.mp3 --model small --language en
"""

import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel


def format_time(seconds: float) -> str:
    """Turn 63.5 into '01:03.50' (minutes:seconds.hundredths)."""
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def transcribe(audio_path: str, model_size: str, language: str | None) -> dict:
    # 1. Load the model.
    #    The first time, the weights are downloaded from Hugging Face and cached
    #    in ~/.cache/huggingface. Later runs load them from disk.
    #    device="cpu" + compute_type="int8" = runs on any Mac, uses less memory.
    print(f"Loading Whisper model '{model_size}' ...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # 2. Run transcription.
    #    word_timestamps=True asks Whisper to also estimate when each word starts/ends.
    #    NOTE: `segments` is a lazy generator - the actual work happens while we
    #    loop over it below, not on this line.
    print(f"Transcribing '{audio_path}' ...")
    segments, info = model.transcribe(
        audio_path,
        language=language,      # None = let Whisper auto-detect the language
        word_timestamps=True,
        beam_size=5,
    )
    print(f"Detected language: {info.language} (probability {info.language_probability:.2f})")
    print(f"Audio duration: {format_time(info.duration)}\n")

    # 3. Collect the results into plain Python lists/dicts (JSON-friendly).
    #    Each loop iteration is where Whisper actually decodes the next segment,
    #    so we print it right away to watch transcription happen live.
    print("=" * 60)
    print("LIVE TRANSCRIPTION (each segment appears as soon as Whisper decodes it):\n")
    segment_list = []
    word_list = []
    for seg in segments:
        progress = 100 * seg.end / info.duration if info.duration else 0
        print(f"[{format_time(seg.start)} → {format_time(seg.end)}]  ({progress:.0f}% of audio done)")
        print(f"  {seg.text.strip()}")
        for w in seg.words or []:
            print(f"     {w.word.strip():<15} → {format_time(w.start)} – {format_time(w.end)}"
                  f"   (p={w.probability:.2f})")
        print(flush=True)

        segment_list.append({
            "id": seg.id,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
        for w in seg.words or []:
            word_list.append({
                "word": w.word.strip(),
                "start": round(w.start, 2),
                "end": round(w.end, 2),
                "probability": round(w.probability, 3),  # Whisper's confidence in this word
                "segment_id": seg.id,                     # which segment the word belongs to
            })

    full_transcript = " ".join(s["text"] for s in segment_list)

    return {
        "audio_file": str(audio_path),
        "model": model_size,
        "language": info.language,
        "duration": round(info.duration, 2),
        "full_transcript": full_transcript,
        "segments": segment_list,
        "words": word_list,
    }


def print_result(result: dict) -> None:
    # Segments and words were already printed live; finish with the full text.
    print("=" * 60)
    print("FULL TRANSCRIPT:\n")
    print(result["full_transcript"])
    print(f"\n{len(result['segments'])} segments, {len(result['words'])} words")


def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file with Whisper.")
    parser.add_argument("audio", help="Path to the audio file (wav, mp3, m4a, ...)")
    parser.add_argument("--model", default="small",
                        help="Whisper model size: tiny, base, small, medium, large-v3 (default: small)")
    parser.add_argument("--language", default=None,
                        help="Language code, e.g. 'en'. Omit to auto-detect.")
    parser.add_argument("--output", default=None,
                        help="Where to save JSON (default: outputs/<audio name>.json)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    result = transcribe(str(audio_path), args.model, args.language)
    print_result(result)

    output_path = Path(args.output) if args.output else Path("outputs") / f"{audio_path.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved JSON to: {output_path}")


if __name__ == "__main__":
    main()
