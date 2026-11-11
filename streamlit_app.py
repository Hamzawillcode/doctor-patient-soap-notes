"""Streamlit front end for clinical documentation. It talks ONLY to the FastAPI backend over HTTP.

Run (with the API already running):   streamlit run streamlit_app.py
"""

import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")
EXAMPLE = Path(__file__).parent / "examples" / "example_transcript_2.txt"

st.set_page_config(page_title="SOAP Notes", page_icon="🩺", layout="wide")
client = httpx.Client(base_url=API, timeout=30)


def api(method: str, path: str, **kwargs):
    """Call the API; show the server's error message instead of a stack trace."""
    try:
        r = client.request(method, path, **kwargs)
    except httpx.ConnectError:
        st.error(f"Cannot reach the API at {API}. Start it with `uvicorn app.main:app`.")
        st.stop()
    if r.is_error:
        st.error(f"{r.status_code}: {r.json().get('detail', r.text) if r.headers.get('content-type') == 'application/json' else r.text}")
        return None
    return r.json() if r.content else {}


# --- sidebar: start a new consultation ------------------------------------------------------------------

with st.sidebar:
    st.header("New consultation")
    audio_tab, text_tab = st.tabs(["Audio", "Transcript"])
    with audio_tab:
        audio = st.file_uploader("Recording", type=["wav", "mp3", "m4a", "flac", "ogg", "webm"])
        st.caption("Runs all 7 stages: Whisper → pyannote → alignment → roles → BioBERT → facts → Qwen3.")
        if st.button("Process audio", disabled=audio is None, type="primary"):
            c = api("POST", "/consultations/audio", files={"file": (audio.name, audio.getvalue(), audio.type)})
            if c:
                st.session_state.selected = c["id"]
                st.rerun()
    with text_tab:
        if st.button("Load example"):
            st.session_state.transcript_text = EXAMPLE.read_text()
        title = st.text_input("Title", "Pasted transcript")
        text = st.text_area("One turn per line", key="transcript_text", height=220,
                            placeholder="DOCTOR: What brings you in today?\nPATIENT: I have had a cough...")
        if st.button("Process transcript", disabled=not text.strip(), type="primary"):
            c = api("POST", "/consultations/transcript", json={"title": title, "text": text})
            if c:
                st.session_state.selected = c["id"]
                st.rerun()


# --- choose a consultation ------------------------------------------------------------------------------

st.title("🩺 Doctor–patient consultation → SOAP note")
consultations = api("GET", "/consultations") or []
if not consultations:
    st.info("No consultations yet. Upload a recording or paste a transcript in the sidebar.")
    st.stop()

ICON = {"queued": "⏳", "running": "⚙️", "completed": "✅", "failed": "❌"}
ids = [c["id"] for c in consultations]
labels = {c["id"]: f"#{c['id']} {ICON[c['status']]} {c['title']}"
                   + ("  · approved" if c["review_status"] == "approved" else "") for c in consultations}
default = ids.index(st.session_state.get("selected")) if st.session_state.get("selected") in ids else 0
selected = st.selectbox("Consultation", ids, index=default, format_func=labels.get)
st.session_state.selected = selected


# --- progress: refreshes itself every 3 s while the pipeline runs ------------------------------------------

def progress_panel():
    c = api("GET", f"/consultations/{selected}")
    if c is None:
        st.stop()
    if c["status"] in ("queued", "running"):
        stage = c["current_stage"] or "waiting for the previous job to finish"
        st.progress(c["stages_done"] / max(c["stages_total"], 1),
                    text=f"{ICON[c['status']]} {c['status'].title()}: stage {c['stages_done'] + 1}/{c['stages_total']} — {stage}")
        st.caption("Large models load in each stage; the first run downloads them. This page updates by itself.")
    elif st.session_state.get("shown_status", {}).get(selected) in ("queued", "running"):
        st.session_state.shown_status[selected] = c["status"]
        st.rerun(scope="app")                    # job just finished: redraw the whole page with results
    st.session_state.setdefault("shown_status", {})[selected] = c["status"]
    return c


first = api("GET", f"/consultations/{selected}")
running = first and first["status"] in ("queued", "running")
c = st.fragment(progress_panel, run_every=3 if running else None)()

if c["status"] == "failed":
    st.error("Pipeline failed")
    st.code(c["error"] or "unknown error")
    st.stop()
if c["status"] != "completed":
    st.stop()


# --- results ----------------------------------------------------------------------------------------------

note_tab, transcript_tab, entities_tab, facts_tab, history_tab = st.tabs(
    ["SOAP note", "Transcript", "Clinical entities", "SOAP facts", "Version history"])

with note_tab:
    g, gen = c["grounding"], c["generation"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Grounding check", "passed" if g["passed"] else f"{len(g['problems'])} flag(s)")
    m2.metric("Fact coverage", f"{g['coverage']:.0%}")
    m3.metric("Note version", f"v{c['latest_version']}")
    m4.metric("Review", c["review_status"])
    st.caption(f"Written by {gen['language_model']} in {gen['seconds']} s ({gen['generated_tokens']} tokens, "
               f"greedy decoding). Every line cites fact IDs [F#] — see the SOAP facts tab.")
    for p in g["problems"]:
        st.warning(f"**{p['check']}** — {p['issue']}" + (f"\n\n`{p['line']}`" if p["line"] else ""))
    if g["facts_not_used"]:
        st.info("Facts not used in the note: " + "; ".join(g["facts_not_used"]))

    approved = c["review_status"] == "approved"
    edited = st.text_area("Note (edit, then save a new version)", c["latest_note"], height=380,
                          disabled=approved, key=f"note_{selected}_{c['latest_version']}")
    left, mid, right = st.columns([2, 2, 1])
    author = left.text_input("Clinician name", key="author", disabled=approved)
    comment = mid.text_input("Change comment (optional)", disabled=approved)
    right.download_button("Download .md", f"# SOAP note — {c['title']}\n\n{edited}\n",
                          file_name=f"consultation_{selected}_soap_note.md")
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Save version", disabled=approved or edited == c["latest_note"] or not author):
        if api("POST", f"/consultations/{selected}/notes",
               json={"content": edited, "author": author, "comment": comment or None}):
            st.toast("Saved new version")
            st.rerun()
    if b2.button("Approve", type="primary", disabled=approved or not author or edited != c["latest_note"],
                 help="Save your edits first; approval locks the note."):
        if api("POST", f"/consultations/{selected}/approve", json={"author": author}):
            st.rerun()
    if approved:
        st.success(f"Approved by {c['approved_by']} at {c['approved_at'][:19].replace('T', ' ')} UTC — locked.")
    elif not author:
        st.caption("Enter your name to save or approve.")

with transcript_tab:
    for t in c["transcript"]:
        role = t.get("role") or t.get("speaker") or "?"
        when = f"{t['start']:.1f}s " if t.get("start") is not None else ""
        with st.chat_message("user" if role == "PATIENT" else "assistant", avatar="🧑" if role == "PATIENT" else "🩺"):
            st.markdown(f"**{role}** {when}  \n{t['text']}")

with entities_tab:
    rows = [{"label": label, "entity": e["text"], "said by": e["speaker"]}
            for label, items in c["entities"].items() for e in items]
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.write("No entities found.")

with facts_tab:
    st.caption("Structured facts built from the BioBERT entities. The language model may only use these.")
    st.json(c["soap_facts"], expanded=1)

with history_tab:
    for v in reversed(api("GET", f"/consultations/{selected}/notes") or []):
        with st.expander(f"v{v['version']} — {v['author']} — {v['created_at'][:19].replace('T', ' ')}"
                         + (f" — {v['comment']}" if v["comment"] else "")):
            st.code(v["content"], language=None)
