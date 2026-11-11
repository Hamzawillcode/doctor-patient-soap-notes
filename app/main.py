"""FastAPI backend: REST API over the doctor-patient -> SOAP note pipeline.

Run:   uvicorn app.main:app --reload        (interactive docs at http://localhost:8000/docs)

    GET    /health                              API + database check
    POST   /consultations/audio                 upload audio  -> 202, pipeline starts in the background
    POST   /consultations/transcript            paste "DOCTOR: ... / PATIENT: ..." text -> 202
    GET    /consultations                       list (newest first)
    GET    /consultations/{id}                  status, progress and all results
    DELETE /consultations/{id}
    GET    /consultations/{id}/notes            every saved version of the SOAP note
    POST   /consultations/{id}/notes            clinician saves an edited note (new version)
    POST   /consultations/{id}/approve          clinician signs off the latest version
"""

import re
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from .database import Consultation, NoteVersion, get_db, init_db, now
from .pipeline import ROOT, mark_interrupted_jobs, run_pipeline, stages_for

UPLOADS = ROOT / "uploads"                      # git-ignored: real patient data never gets committed
AUDIO_TYPES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
MAX_UPLOAD_MB = 200


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()                    # create tables if they don't exist yet
    mark_interrupted_jobs()
    UPLOADS.mkdir(exist_ok=True)
    yield


app = FastAPI(title="Doctor-Patient SOAP Notes API", version="1.0", lifespan=lifespan)


# --- response / request shapes (Pydantic validates and documents them in /docs) ----------------------------

class ConsultationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    source_type: str
    status: str
    current_stage: str | None
    stages_done: int
    stages_total: int
    review_status: str
    created_at: datetime


class ConsultationDetail(ConsultationSummary):
    error: str | None
    transcript: list | None
    entities: dict | None
    soap_facts: list | None
    generated_note: str | None
    grounding: dict | None
    generation: dict | None
    approved_by: str | None
    approved_at: datetime | None
    latest_note: str | None = None
    latest_version: int | None = None


class NoteVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    version: int
    content: str
    author: str
    comment: str | None
    created_at: datetime


class TranscriptIn(BaseModel):
    title: str = Field(default="Pasted transcript", max_length=200)
    text: str = Field(min_length=10, description="One turn per line: 'DOCTOR: ...' or 'PATIENT: ...'")


class NoteIn(BaseModel):
    content: str = Field(min_length=1)
    author: str = Field(min_length=1, max_length=100)
    comment: str | None = Field(default=None, max_length=500)


class ApproveIn(BaseModel):
    author: str = Field(min_length=1, max_length=100)


# --- helpers ----------------------------------------------------------------------------------------------

def get_or_404(db: Session, consultation_id: int) -> Consultation:
    c = db.get(Consultation, consultation_id)
    if c is None:
        raise HTTPException(404, f"Consultation {consultation_id} not found")
    return c


def start_job(db: Session, background: BackgroundTasks, c: Consultation, save_input) -> Consultation:
    """Insert the row (to get an id), write the input file named after it, then schedule the pipeline."""
    db.add(c)
    db.flush()                                          # assigns c.id without committing
    path = save_input(f"consult_{c.id}")
    c.input_path = str(path.relative_to(ROOT))
    c.stages_total = len(stages_for(c.source_type, c.input_path, path.stem))
    db.commit()
    # Runs after the 202 response is sent; it is an async function, so it runs on the server's event loop.
    background.add_task(run_pipeline, c.id, c.source_type, c.input_path)
    return c


def detail(c: Consultation) -> ConsultationDetail:
    out = ConsultationDetail.model_validate(c)
    if c.versions:
        out.latest_note, out.latest_version = c.versions[-1].content, c.versions[-1].version
    return out


# --- endpoints --------------------------------------------------------------------------------------------

@app.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(text("select 1"))
    return {"status": "ok", "database": "ok"}


@app.post("/consultations/audio", status_code=202, response_model=ConsultationSummary)
def upload_audio(file: UploadFile, background: BackgroundTasks, db: Session = Depends(get_db)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in AUDIO_TYPES:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(AUDIO_TYPES))}")
    if file.size is not None and file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")

    def save(stem: str) -> Path:
        path = UPLOADS / f"{stem}{ext}"
        with open(path, "wb") as out:
            shutil.copyfileobj(file.file, out)          # streams in chunks, never the whole file in memory
        return path

    c = Consultation(title=Path(file.filename).name[:200], source_type="audio", input_path="")
    return start_job(db, background, c, save)


@app.post("/consultations/transcript", status_code=202, response_model=ConsultationSummary)
def upload_transcript(body: TranscriptIn, background: BackgroundTasks, db: Session = Depends(get_db)):
    turns = [line for line in body.text.splitlines() if line.strip()]
    bad = [line for line in turns if not re.match(r"^\s*(DOCTOR|PATIENT)\s*:", line, re.IGNORECASE)]
    if not turns or bad:
        raise HTTPException(422, f"Every line must start with 'DOCTOR:' or 'PATIENT:'. First bad line: {bad[:1]}")

    def save(stem: str) -> Path:
        path = UPLOADS / f"{stem}.txt"
        path.write_text("\n".join(turns) + "\n")
        return path

    c = Consultation(title=body.title, source_type="transcript", input_path="")
    return start_job(db, background, c, save)


@app.get("/consultations", response_model=list[ConsultationSummary])
def list_consultations(limit: int = 50, db: Session = Depends(get_db)):
    return db.scalars(select(Consultation).order_by(Consultation.id.desc()).limit(min(limit, 200))).all()


@app.get("/consultations/{consultation_id}", response_model=ConsultationDetail)
def get_consultation(consultation_id: int, db: Session = Depends(get_db)):
    return detail(get_or_404(db, consultation_id))


@app.delete("/consultations/{consultation_id}", status_code=204)
def delete_consultation(consultation_id: int, db: Session = Depends(get_db)):
    c = get_or_404(db, consultation_id)
    if c.status in ("queued", "running"):
        raise HTTPException(409, "Pipeline is still running; delete it after it finishes")
    (ROOT / c.input_path).unlink(missing_ok=True)
    db.delete(c)
    db.commit()


@app.get("/consultations/{consultation_id}/notes", response_model=list[NoteVersionOut])
def list_note_versions(consultation_id: int, db: Session = Depends(get_db)):
    return get_or_404(db, consultation_id).versions


@app.post("/consultations/{consultation_id}/notes", status_code=201, response_model=NoteVersionOut)
def save_note_version(consultation_id: int, body: NoteIn, db: Session = Depends(get_db)):
    c = get_or_404(db, consultation_id)
    if c.status != "completed":
        raise HTTPException(409, "The SOAP note can be edited once the pipeline has completed")
    if c.review_status == "approved":
        raise HTTPException(409, "This note is approved; it can no longer be edited")
    last = db.scalar(select(func.max(NoteVersion.version)).where(NoteVersion.consultation_id == c.id)) or 0
    v = NoteVersion(consultation_id=c.id, version=last + 1, content=body.content,
                    author=body.author, comment=body.comment)
    db.add(v)
    db.commit()
    return v


@app.post("/consultations/{consultation_id}/approve", response_model=ConsultationDetail)
def approve_note(consultation_id: int, body: ApproveIn, db: Session = Depends(get_db)):
    c = get_or_404(db, consultation_id)
    if c.status != "completed":
        raise HTTPException(409, "Nothing to approve until the pipeline has completed")
    c.review_status, c.approved_by, c.approved_at = "approved", body.author, now()
    db.commit()
    return detail(c)
