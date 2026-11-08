# Doctor–Patient Conversation → SOAP Notes

A learning project that will convert doctor–patient conversation audio into structured SOAP notes,
built one stage at a time.

| Stage | Status | Script | Notes |
|---|---|---|---|
| 1. Speech-to-text (faster-whisper) | done | `transcribe.py` | `docs/01_speech_to_text_notes.pdf` |
| 2. Speaker diarization (pyannote.audio) | done | `diarize.py` | `docs/02_speaker_diarization_notes.pdf` |
| 3. Timestamp alignment (max temporal overlap) | done | `align.py` | `docs/03_timestamp_alignment_notes.pdf` |
| 4. Doctor / patient role identification (rule-based) | done | `identify_roles.py` | `docs/04_role_identification_notes.pdf` |
| 5. Clinical NER | not started | | |
| 6. SOAP note generation | not started | | |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt                 # Stage 1: faster-whisper
pip install -r requirements-diarization.txt     # Stage 2: pyannote.audio (~700 MB, includes PyTorch)
```

Stage 2 uses the gated model `pyannote/speaker-diarization-community-1`:
accept its conditions on Hugging Face, create a Read token, then run `hf auth login`.

## Run

```bash
python transcribe.py audio/sample_consultation.wav     # -> outputs/sample_consultation.json
python diarize.py audio/sample_consultation.wav        # -> outputs/sample_consultation_diarization.json
python align.py audio/sample_consultation.wav         # -> outputs/sample_consultation_aligned.json
python identify_roles.py audio/sample_consultation.wav  # -> outputs/sample_consultation_roles.json
```

`audio/sample_consultation.wav` and `audio/sample_consultation_2.wav` are synthetic two-voice clips made with
macOS `say` (clip 2 is harder: the patient speaks first and pauses are only 0.15 s).
Other files in `audio/` and `outputs/` are git-ignored so real patient recordings are never committed.
