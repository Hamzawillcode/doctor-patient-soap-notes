"""
Step 2 of the pipeline: Speaker Diarization with pyannote.audio ("who spoke when?").

    audio file  ->  pyannote.audio  ->  anonymous speaker labels + time intervals

This is completely separate from transcribe.py. It does NOT look at any text.
SPEAKER_00 / SPEAKER_01 are anonymous labels - they do NOT mean doctor or patient.

Usage:
    python diarize.py audio/sample_consultation.wav
    python diarize.py audio/sample_consultation.wav --num-speakers 2
    python diarize.py audio/sample_consultation.wav --device mps
"""

import argparse
import json
import time
from pathlib import Path

import torch
from faster_whisper.audio import decode_audio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

PIPELINE_NAME = "pyannote/speaker-diarization-community-1"
SAMPLE_RATE = 16000


def format_time(seconds: float) -> str:
    """Turn 63.5 into '01:03.50' (minutes:seconds.hundredths)."""
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def annotation_to_list(annotation) -> list[dict]:
    """Convert a pyannote Annotation (its result object) into plain JSON-friendly dicts."""
    turns = []
    for segment, _track, speaker in annotation.itertracks(yield_label=True):
        turns.append({
            "speaker": speaker,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
        })
    return turns


def diarize(audio_path: str, device: str, num_speakers: int | None) -> dict:
    # 1. Load the pretrained pipeline.
    #    First run: downloads the model weights (~35 MB) from Hugging Face.
    #    The model is "gated", so it uses the token saved by `hf auth login`.
    print(f"Loading pipeline '{PIPELINE_NAME}' ...")
    try:
        pipeline = Pipeline.from_pretrained(PIPELINE_NAME)
    except Exception as e:
        raise SystemExit(
            f"\nCould not load the pipeline: {e}\n\n"
            "Check that you:\n"
            f"  1. accepted the conditions at https://huggingface.co/{PIPELINE_NAME}\n"
            "  2. logged in with:  .venv/bin/hf auth login\n"
        )
    if pipeline is None:
        raise SystemExit("Pipeline could not be loaded - see the Hugging Face steps above.")
    pipeline.to(torch.device(device))

    # 2. Load the audio ourselves (with PyAV, same decoder Whisper uses) instead of
    #    letting pyannote read the file. This means no ffmpeg is needed, and both
    #    stages see exactly the same 16 kHz mono samples.
    #    pyannote expects a tensor shaped (channels, samples) -> (1, N).
    audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    duration = waveform.shape[1] / SAMPLE_RATE
    print(f"Audio duration: {format_time(duration)}  |  device: {device}\n")

    # 3. Run diarization. The ProgressHook shows each internal step live:
    #    segmentation -> speaker counting -> embeddings -> clustering.
    #    num_speakers is optional; if given, clustering is forced to that many speakers.
    start_time = time.time()
    with ProgressHook() as hook:
        output = pipeline(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE},
            num_speakers=num_speakers,
            hook=hook,
        )
    elapsed = time.time() - start_time

    # 4. pyannote 4.x returns two versions of the result:
    #    - speaker_diarization: regular - overlapping speech can have 2 speakers at once
    #    - exclusive_speaker_diarization: at most ONE speaker at any moment
    #      (simpler to match with transcription timestamps later)
    speakers = annotation_to_list(output.speaker_diarization)
    exclusive = annotation_to_list(output.exclusive_speaker_diarization)

    return {
        "audio_file": str(audio_path),
        "pipeline": PIPELINE_NAME,
        "duration": round(duration, 2),
        "processing_seconds": round(elapsed, 2),
        "num_speakers_found": len({t["speaker"] for t in speakers}),
        "speakers": speakers,
        "exclusive_speakers": exclusive,
    }


def print_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print("WHO SPOKE WHEN (speaker turns):\n")
    for t in result["speakers"]:
        print(f"[{format_time(t['start'])} → {format_time(t['end'])}]  {t['speaker']}")

    print("\n" + "=" * 60)
    print("TOTAL SPEAKING TIME PER SPEAKER:\n")
    totals = {}
    for t in result["speakers"]:
        totals[t["speaker"]] = totals.get(t["speaker"], 0) + (t["end"] - t["start"])
    for speaker, seconds in sorted(totals.items()):
        print(f"{speaker}: {seconds:.2f} s")

    print(f"\nSpeakers found: {result['num_speakers_found']}"
          f"   |   processing time: {result['processing_seconds']} s")


def main():
    parser = argparse.ArgumentParser(description="Speaker diarization with pyannote.audio.")
    parser.add_argument("audio", help="Path to the audio file (wav, mp3, m4a, ...)")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Exact number of speakers, if known (e.g. 2). Omit to let pyannote decide.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                        help="cpu (default), mps (Apple Silicon GPU) or cuda (NVIDIA GPU)")
    parser.add_argument("--output", default=None,
                        help="Where to save JSON (default: outputs/<audio name>_diarization.json)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    result = diarize(str(audio_path), args.device, args.num_speakers)
    print_result(result)

    output_path = (Path(args.output) if args.output
                   else Path("outputs") / f"{audio_path.stem}_diarization.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved JSON to: {output_path}")


if __name__ == "__main__":
    main()
