# Doctor–Patient Conversation → SOAP Notes

A learning project that converts doctor–patient conversation audio into structured SOAP notes,
built one stage at a time, and served as a web app (FastAPI + Streamlit + PostgreSQL).

| Stage | Status | Script | Notes |
|---|---|---|---|
| 1. Speech-to-text (faster-whisper) | done | `transcribe.py` | `docs/01_speech_to_text_notes.pdf` |
| 2. Speaker diarization (pyannote.audio) | done | `diarize.py` | `docs/02_speaker_diarization_notes.pdf` |
| 3. Timestamp alignment (max temporal overlap) | done | `align.py` | `docs/03_timestamp_alignment_notes.pdf` |
| 4. Doctor / patient role identification (rule-based) | done | `identify_roles.py` | `docs/04_role_identification_notes.pdf` |
| 5. Clinical NER (BioBERT) | done | `clinical_extraction.py` | `docs/05_clinical_ner_biobert_notes.pdf` |
| 6a. SOAP structured facts (rules, grounded) | done | `build_soap_facts.py` | `docs/06_soap_generation_notes.pdf` |
| 6b/6c. SOAP note writing (pretrained Qwen3-4B-Instruct, local) + grounding check | done | `generate_soap_note.py` | `docs/06_soap_generation_notes.pdf` |
| 7. Full-stack app: REST API, async pipeline jobs, PostgreSQL, clinician review UI | done | `app/`, `streamlit_app.py` | `docs/07_fullstack_app_notes.pdf` |

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
python generate_soap_note.py audio/sample_consultation_2.wav   # ~8 GB model download on first run
python generate_soap_note.py --facts outputs/example_transcript_2_soap_facts.json
```

`audio/sample_consultation.wav` and `audio/sample_consultation_2.wav` are synthetic two-voice clips made with
macOS `say` (clip 2 is harder: the patient speaks first and pauses are only 0.15 s).
Other files in `audio/` and `outputs/` are git-ignored so real patient recordings are never committed.

## Web app (Stage 7)

```
Streamlit UI  --HTTP-->  FastAPI (app/main.py)  --SQLAlchemy-->  PostgreSQL (consultations, note_versions)
                               |
                               └─ background job (app/pipeline.py): runs the 7 stage scripts as async child processes
```

```bash
pip install -r requirements-app.txt
brew install postgresql@16 && brew services start postgresql@16
/opt/homebrew/opt/postgresql@16/bin/createdb soap_notes      # tables are created on first API start

uvicorn app.main:app --port 8000          # REST API; interactive docs at http://localhost:8000/docs
streamlit run streamlit_app.py            # UI at http://localhost:8501 (second terminal)
```

Set `DATABASE_URL` (default `postgresql+psycopg://localhost/soap_notes`) or `API_URL` (default `http://localhost:8000`)
to point elsewhere. Upload a recording (all 7 stages) or paste a `DOCTOR: ... / PATIENT: ...` transcript (NER → facts → LM),
watch progress live, then edit, version and approve the SOAP note. Jobs run one at a time (the models need ~8 GB).

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API + database check |
| POST | `/consultations/audio` | upload audio → `202`, pipeline runs in the background |
| POST | `/consultations/transcript` | JSON `{title, text}` → `202` |
| GET | `/consultations`, `/consultations/{id}` | list / status, progress and all results |
| DELETE | `/consultations/{id}` | remove a finished consultation |
| GET, POST | `/consultations/{id}/notes` | note version history / save a clinician edit |
| POST | `/consultations/{id}/approve` | sign off and lock the note |

Uploads go to `uploads/` and job logs to `logs/` (both git-ignored). There is no authentication: run it locally only.

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
