"""PostgreSQL connection + the two tables the app stores.

    consultations   one row per uploaded audio file / pasted transcript, with pipeline status and results
    note_versions   every saved version of the SOAP note (v1 = written by the LM, v2+ = clinician edits)

Results that are naturally nested (transcript turns, entities, facts, grounding report) are stored in
JSONB columns: PostgreSQL keeps them as queryable JSON, so we don't need a table per pipeline stage.
"""

import os
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

# Override with e.g. DATABASE_URL=postgresql+psycopg://user:password@host:5432/soap_notes
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://localhost/soap_notes")

# timezone=utc: PostgreSQL returns timestamps in UTC instead of the server machine's local zone.
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args={"options": "-c timezone=utc"})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    # Store every datetime as PostgreSQL "timestamptz" so times round-trip in UTC, whatever the server's timezone.
    type_annotation_map = {datetime: DateTime(timezone=True)}


class Consultation(Base):
    __tablename__ = "consultations"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    source_type: Mapped[str] = mapped_column(String(20))            # "audio" or "transcript"
    input_path: Mapped[str] = mapped_column(String(500))

    # Pipeline progress
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)  # queued/running/completed/failed
    current_stage: Mapped[str | None] = mapped_column(String(50))
    stages_done: Mapped[int] = mapped_column(default=0)
    stages_total: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(Text)

    # Results (filled when the pipeline finishes)
    transcript: Mapped[list | None] = mapped_column(JSONB)          # [{role, speaker, start, end, text}]
    entities: Mapped[dict | None] = mapped_column(JSONB)            # {label: [{text, speaker}]}
    soap_facts: Mapped[list | None] = mapped_column(JSONB)
    generated_note: Mapped[str | None] = mapped_column(Text)        # raw LM output, never overwritten
    grounding: Mapped[dict | None] = mapped_column(JSONB)
    generation: Mapped[dict | None] = mapped_column(JSONB)          # model name, tokens, seconds

    # Clinician review
    review_status: Mapped[str] = mapped_column(String(20), default="draft")   # draft / approved
    approved_by: Mapped[str | None] = mapped_column(String(100))
    approved_at: Mapped[datetime | None]

    created_at: Mapped[datetime] = mapped_column(default=now)
    updated_at: Mapped[datetime] = mapped_column(default=now, onupdate=now)

    versions: Mapped[list["NoteVersion"]] = relationship(
        back_populates="consultation", cascade="all, delete-orphan", order_by="NoteVersion.version")


class NoteVersion(Base):
    __tablename__ = "note_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    consultation_id: Mapped[int] = mapped_column(ForeignKey("consultations.id", ondelete="CASCADE"), index=True)
    version: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    author: Mapped[str] = mapped_column(String(100))
    comment: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(default=now)

    consultation: Mapped[Consultation] = relationship(back_populates="versions")


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db():
    """FastAPI dependency: one session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
