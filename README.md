# Doctor–Patient Conversation → SOAP Notes

A learning project that will convert doctor–patient conversation audio into structured SOAP notes,
built one stage at a time.

| Stage | Status | Script | Notes |
|---|---|---|---|
| 1. Speech-to-text (faster-whisper) | done | `transcribe.py` | `docs/01_speech_to_text_notes.pdf` |
| 2. Speaker diarization (pyannote.audio) | done | `diarize.py` | `docs/02_speaker_diarization_notes.pdf` |
| 3. Timestamp alignment (max temporal overlap) | done | `align.py` | `docs/03_timestamp_alignment_notes.pdf` |
| 4. Doctor / patient role identification (rule-based) | done | `identify_roles.py` | `docs/04_role_identification_notes.pdf` |
| 5. Clinical NER (BioBERT) | done | `clinical_extraction.py` | `docs/05_clinical_ner_biobert_notes.pdf` |
| 6a. SOAP structured facts (rules, grounded) | done | `build_soap_facts.py` | `docs/06_soap_generation_notes.pdf` |
| 6b. SOAP note writing (pretrained Transformer LM) + grounding check | next | | |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt                 # Stage 1: faster-whisper
pip install -r requirements-diarization.txt     # Stage 2: pyannote.audio (~700 MB, includes PyTorch)
pip install -r requirements-ner.txt             # Stage 5: transformers (BioBERT models ~860 MB download on first run)
```

Stage 2 uses the gated model `pyannote/speaker-diarization-community-1`:
accept its conditions on Hugging Face, create a Read token, then run `hf auth login`.

## Run

```bash
python transcribe.py audio/sample_consultation.wav     # -> outputs/sample_consultation.json
python diarize.py audio/sample_consultation.wav        # -> outputs/sample_consultation_diarization.json
python align.py audio/sample_consultation.wav         # -> outputs/sample_consultation_aligned.json
python identify_roles.py audio/sample_consultation.wav  # -> outputs/sample_consultation_roles.json
python clinical_extraction.py audio/sample_consultation.wav   # -> outputs/sample_consultation_clinical_entities.json
python clinical_extraction.py --transcript examples/example_transcript.txt
python clinical_extraction.py --transcript examples/example_transcript.txt --models bc5cdr   # public BioBERT checkpoints
python build_soap_facts.py audio/sample_consultation.wav   # -> outputs/sample_consultation_soap_facts.json
python build_soap_facts.py --entities outputs/example_transcript_2_clinical_entities.json
```

`audio/sample_consultation.wav` and `audio/sample_consultation_2.wav` are synthetic two-voice clips made with
macOS `say` (clip 2 is harder: the patient speaks first and pauses are only 0.15 s).
Other files in `audio/` and `outputs/` are git-ignored so real patient recordings are never committed.

## Stage 5: fine-tuning BioBERT on MACCROBAT2018

`clinical_extraction.py` uses our fine-tuned model by default (`--models maccrobat`). To recreate it:

```bash
mkdir -p data/maccrobat
curl -L -o data/maccrobat/data.jsonl \
  https://huggingface.co/datasets/ktgiahieu/maccrobat2018_2020/resolve/main/data.jsonl
python train_biobert_ner.py          # ~12 min on Apple M4 (mps); saves models/biobert-maccrobat-ner/
```

Dataset: MACCROBAT2018 — Caufield, J. Harry (2019), figshare, https://doi.org/10.6084/m9.figshare.9764942 (CC-BY-4.0),
via the Hugging Face copy `ktgiahieu/maccrobat2018_2020` (duplicates removed during training).
Test results are in `outputs/biobert_maccrobat_training_metrics.json`.
