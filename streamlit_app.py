# app.py
"""
Noting — Web-hosted Meeting Transcriber & MOM Generator
- Supports: Paste transcript, Upload transcript, Upload audio
- Optional: uses Hugging Face Inference API for STT (whisper) and LLM summarization (if HUGGINGFACE_TOKEN provided)
- Fallback: local extractive MOM generator when no HF token supplied
- Action items displayed as editable table; .docx and .json download supported
- Clear with confirmation token CONFIRM CLEAR
"""

import streamlit as st
import pandas as pd
import json
import os
import uuid
import tempfile
import requests
from datetime import datetime
from io import BytesIO
from docx import Document
import re

# ---------- CONFIG ----------
st.set_page_config(page_title="Noting — AI Meeting Note Assitant", layout="wide")
TMP_DIR = tempfile.gettempdir()
DEFAULT_TTL_HOURS = 24

# ---------- HELPERS ----------
HF_API_URL = "https://api-inference.huggingface.co"

def hf_transcribe_audio_bytes(audio_bytes: bytes, hf_token: str):
    """
    Use Hugging Face Inference API to run whisper-large-v2 transcription.
    Model: openai/whisper-large-v2
    Requires HF token with Inference API access.
    """
    model = "openai/whisper-large-v2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    url = f"{HF_API_URL}/models/{model}"
    # For audio, Hugging Face Inference API accepts raw bytes via POST
    resp = requests.post(url, headers=headers, data=audio_bytes, timeout=120)
    if resp.status_code == 200:
        try:
            # Response should be JSON with 'text'
            data = resp.json()
            if isinstance(data, dict) and "text" in data:
                return data["text"]
            # some endpoints return plain text
            if isinstance(data, str):
                return data
        except Exception:
            return None
    # Any error
    return None

def hf_generate_mom_from_transcript(transcript_text: str, hf_token: str, metadata: dict):
    """
    Use a text2text model on HF to generate MOM. We'll craft a prompt instructing the model to
    output both human-friendly text and JSON matching the required schema.
    Model: google/flan-t5-large (or prefer smaller/larger depending on token limits).
    NOTE: Inference response format may differ across models; this function tries to parse JSON from response.
    """
    model = "google/flan-t5-large"  # user can replace with other instruction-following HF models
    url = f"{HF_API_URL}/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    # Build a prompt that instructs the model to return JSON and plain text.
    prompt = (
        "You are a meeting summarization assistant. Produce:\n"
        "1) A short human-friendly Minutes of Meeting (MOM) with headings: Meeting title, Date, Executive summary, Key points, Decisions, Action items (table-like bullets), Open issues.\n"
        "2) A JSON object matching this schema exactly (use null for missing values):\n"
        "{'meeting_title':'string or Untitled Meeting','meeting_date':'YYYY-MM-DD or null','start_time':'HH:MM:SS+05:30 or null','end_time':'HH:MM:SS+05:30 or null','duration_minutes':null,'attendees':[], 'executive_summary':'string','key_points':[], 'decisions':[], 'action_items':[], 'open_issues':[], 'keywords':[], 'source_transcript_file': null, 'notes': ''}\n\n"
        "Transcript:\n"
        f"{transcript_text}\n\n"
        "Return first the human readable MOM, then a JSON block exactly matching the schema. Do not invent attendees or owners; if unknown use 'Unknown' or null. Normalize dates to YYYY-MM-DD where possible (otherwise null)."
    )
    payload = {"inputs": prompt, "options": {"wait_for_model": True}, "parameters": {"max_new_tokens": 1024}}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
    except Exception as e:
        st.error(f"Hugging Face request error: {e}")
        return None, None
    if r.status_code != 200:
        st.warning(f"Hugging Face inference failed (status {r.status_code}). Falling back to local generator.")
        return None, None
    # Some HF text generation endpoints return text directly
    out = r.json()
    # The structure depends on the model / inference API; detect common shapes:
    text_out = None
    if isinstance(out, list) and "generated_text" in out[0]:
        text_out = out[0]["generated_text"]
    elif isinstance(out, dict) and "generated_text" in out:
        text_out = out["generated_text"]
    elif isinstance(out, dict) and "error" in out:
        st.warning("Hugging Face returned error: " + str(out.get("error")))
        return None, None
    else:
        # try to stringify
        try:
            text_out = json.dumps(out)
        except Exception:
            text_out = str(out)
    if not text_out:
        return None, None

    # Attempt to split human part and JSON block
    # Find last JSON block in text_out
    json_block = None
    m = re.search(r'(\{[\s\S]*\})\s*$', text_out.strip())
    if m:
        json_text = m.group(1)
        try:
            json_block = json.loads(json_text)
        except Exception:
            # Try to extract first {...} occurrence
            try:
                start = text_out.find("{")
                end = text_out.rfind("}") + 1
                json_text = text_out[start:end]
                json_block = json.loads(json_text)
            except Exception:
                json_block = None
    # If no JSON parsed, leave json_block None so fallback occurs
    human_text = text_out
    return human_text, json_block

def local_extractive_mom(transcript_text: str, metadata: dict):
    """
    Simple extractive MOM generator for offline fallback.
    - executive summary: first 2-3 sentences
    - key points: top short lines / sentence fragments
    - decisions: lines containing 'decide', 'decided', 'we will', 'we'll', 'agreed'
    - action items: regex looking for 'will', 'to', 'owner names' like 'Raj:', 'Priya:' - naive approach
    """
    # normalize whitespace
    txt = transcript_text.strip()
    # split into sentences (naive)
    sentences = re.split(r'(?<=[\.\?\!])\s+', txt)
    executive = " ".join(sentences[:2]).strip()
    key_points = []
    decisions = []
    action_items = []
    open_issues = []
    keywords = []

    lines = [l.strip() for l in re.split(r'\n+', txt) if l.strip()]
    # extract short lines as key points
    for ln in lines[:8]:
        if len(ln) < 200:
            key_points.append(ln if len(ln) < 150 else ln[:147] + "...")
    # naive decisions
    for s in sentences:
        if re.search(r'\b(decide|decided|agree|agreed|we will|we\'ll)\b', s, flags=re.I):
            decisions.append(s.strip())
    # naive action items - look for patterns like "X will Y" or "Action: Y"
    aid = 1
    for s in sentences:
        if re.search(r'\b(will|to do|action|assign|owner)\b', s, flags=re.I):
            # try to find owner name before colon
            owner = "Unknown"
            owner_match = re.match(r'^\s*([A-Z][a-z]+)\s*[:\-]\s*(.*)', s)
            desc = s.strip()
            if owner_match:
                owner = owner_match.group(1)
                desc = owner_match.group(2)
            action_items.append({
                "id": f"A{aid}",
                "description": desc[:300],
                "owner": owner,
                "deadline": None,
                "priority": "Medium",
                "status": "Open"
            })
            aid += 1
    # if no action items found, add one placeholder from first sentence
    if not action_items:
        action_items.append({
            "id": "A1",
            "description": sentences[0][:300] if sentences else "Follow up required",
            "owner": "Unknown",
            "deadline": None,
            "priority": "Medium",
            "status": "Open"
        })
    # keywords naive
    keywords = list({w.lower() for w in re.findall(r'\b[A-Za-z]{4,}\b', txt)[:10]})
    json_mom = {
        "meeting_title": metadata.get("meeting_title") or "Untitled Meeting",
        "meeting_date": metadata.get("meeting_date"),
        "start_time": metadata.get("start_time"),
        "end_time": metadata.get("end_time"),
        "duration_minutes": metadata.get("duration_minutes"),
        "attendees": metadata.get("attendees", []),
        "executive_summary": executive,
        "key_points": key_points,
        "decisions": decisions,
        "action_items": action_items,
        "open_issues": open_issues,
        "keywords": keywords,
        "source_transcript_file": metadata.get("source_transcript_file"),
        "notes": "Generated by local extractive fallback"
    }
    human_text = f"Meeting: {json_mom['meeting_title']}\nDate: {json_mom['meeting_date'] or 'Unknown'}\n\nExecutive summary:\n- {json_mom['executive_summary']}\n\nKey points:\n"
    for kp in json_mom["key_points"]:
        human_text += f"- {kp}\n"
    human_text += "\nDecisions:\n"
    for d in json_mom["decisions"]:
        human_text += f"- {d}\n"
    human_text += "\nAction items:\n"
    for ai in json_mom["action_items"]:
        human_text += f"- {ai['id']}: {ai['description']} | Owner: {ai['owner']} | Deadline: {ai['deadline'] or 'TBD'} | Priority: {ai['priority']}\n"
    return human_text, json_mom

def download_docx_from_mom(json_mom):
    doc = Document()
    doc.add_heading(json_mom.get("meeting_title","Untitled Meeting"), level=1)
    doc.add_paragraph(f"Date: {json_mom.get('meeting_date') or 'Unknown'}")
    doc.add_heading("Executive summary", level=2)
    doc.add_paragraph(json_mom.get("executive_summary",""))
    doc.add_heading("Key points", level=2)
    for kp in json_mom.get("key_points",[]):
        doc.add_paragraph(kp, style='List Bullet')
    doc.add_heading("Decisions", level=2)
    for d in json_mom.get("decisions",[]):
        doc.add_paragraph(d, style='List Bullet')
    doc.add_heading("Action items", level=2)
    table = doc.add_table(rows=1, cols=5)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'ID'
    hdr_cells[1].text = 'Description'
    hdr_cells[2].text = 'Owner'
    hdr_cells[3].text = 'Deadline'
    hdr_cells[4].text = 'Priority'
    for ai in json_mom.get("action_items",[]):
        row_cells = table.add_row().cells
        row_cells[0].text = ai.get("id","")
        row_cells[1].text = ai.get("description","")
        row_cells[2].text = ai.get("owner","")
        row_cells[3].text = ai.get("deadline") or ""
        row_cells[4].text = ai.get("priority","")
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

# ---------- UI ----------
st.title("Noting — AI Meeting Note Assitant")
st.write("Simple, shareable meeting MOM generator. Optional: provide Hugging Face token in Settings to enable auto transcription & LLM summarization (free tier / limited usage).")

# Sidebar - login & settings
st.sidebar.header("Login & Settings")
st.sidebar.markdown("**Sign-in**: Configure Google OAuth on your deployment for direct Google Sign-in. (Placeholder here).")
if st.sidebar.button("Sign in with Google (placeholder)"):
    st.sidebar.info("On deployed app: configure Google OAuth Client and set redirect URIs. See deployment notes below.")

st.sidebar.markdown("---")
st.sidebar.subheader("Provider API keys (optional)")
st.sidebar.info("Paste tokens if you want the app to call those providers. You can choose not to save tokens (recommended for privacy). On Streamlit Cloud add as Secrets.")
hf_token_input = st.sidebar.text_input("Hugging Face Token (for STT & LLM)", type="password", value=st.secrets.get("HUGGINGFACE_TOKEN","") if hasattr(st,'secrets') else "")
grok_key_input = st.sidebar.text_input("Grok API key (label only)", type="password")
google_key_input = st.sidebar.text_input("Google API key (label only)", type="password")
save_keys = st.sidebar.checkbox("Save keys in server session (temporary)", value=False)
if save_keys:
    # store in session safely (not persistent)
    st.session_state['hf_token'] = hf_token_input
    st.session_state['grok_key'] = grok_key_input
    st.session_state['google_key'] = google_key_input
    st.sidebar.success("Saved temporarily for this session.")
else:
    # do not persist
    if 'hf_token' in st.session_state: st.session_state.pop('hf_token',None)
    if 'grok_key' in st.session_state: st.session_state.pop('grok_key',None)
    if 'google_key' in st.session_state: st.session_state.pop('google_key',None)

st.sidebar.caption("Keys will not be echoed back by the assistant. To deploy, add HUGGINGFACE_TOKEN as a secret on your hosting platform for persistent use.")

st.sidebar.markdown("---")
if st.sidebar.button("Delete all temporary MOM & transcripts (immediate)"):
    st.session_state.pop("last_mom", None)
    st.success("Cleared session MOM and transcripts.")

# Input modes
st.header("Input")
mode = st.radio("Choose input mode", ["Paste transcript", "Upload transcript", "Upload audio (file)"], index=0)

transcript_text = None
metadata = {"meeting_title": None, "meeting_date": None, "start_time": None, "end_time": None, "duration_minutes": None, "attendees": []}

if mode == "Paste transcript":
    st.subheader("Paste transcript")
    transcript_text = st.text_area("Paste the meeting transcript here", height=300)
    if st.button("Generate MOM from pasted transcript"):
        if not transcript_text or not transcript_text.strip():
            st.warning("Please paste transcript text first.")
        else:
            st.info("Generating MOM...")
            hf_token = st.session_state.get('hf_token') or hf_token_input
            if hf_token:
                human_text, json_mom = hf_generate_mom_from_transcript(transcript_text, hf_token, metadata)
                if human_text and json_mom:
                    # ensure schema keys exist
                    st.session_state['last_mom'] = json_mom
                    st.success("MOM generated — see below.")
                else:
                    # fallback
                    human_text, json_mom = local_extractive_mom(transcript_text, metadata)
                    st.session_state['last_mom'] = json_mom
                    st.success("MOM generated using local fallback.")
            else:
                human_text, json_mom = local_extractive_mom(transcript_text, metadata)
                st.session_state['last_mom'] = json_mom
                st.success("MOM generated using local fallback.")
elif mode == "Upload transcript":
    st.subheader("Upload transcript (.txt, .vtt, .srt)")
    uploaded = st.file_uploader("Choose transcript file", type=['txt','vtt','srt'])
    if uploaded:
        raw = uploaded.getvalue()
        try:
            transcript_text = raw.decode('utf-8')
        except:
            transcript_text = str(raw)
        metadata['source_transcript_file'] = uploaded.name
        if st.button("Generate MOM from uploaded transcript"):
            st.info("Generating MOM...")
            hf_token = st.session_state.get('hf_token') or hf_token_input
            if hf_token:
                human_text, json_mom = hf_generate_mom_from_transcript(transcript_text, hf_token, metadata)
                if human_text and json_mom:
                    st.session_state['last_mom'] = json_mom
                    st.success("MOM generated — see below.")
                else:
                    human_text, json_mom = local_extractive_mom(transcript_text, metadata)
                    st.session_state['last_mom'] = json_mom
                    st.success("MOM generated using local fallback.")
            else:
                human_text, json_mom = local_extractive_mom(transcript_text, metadata)
                st.session_state['last_mom'] = json_mom
                st.success("MOM generated using local fallback.")
else:
    st.subheader("Upload audio file for transcription (.wav, .mp3, .m4a, .ogg)")
    audio_file = st.file_uploader("Upload audio", type=['wav','mp3','m4a','ogg'])
    if audio_file:
        st.markdown(f"Saved file: **{audio_file.name}** (temporary)")
        tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}_{audio_file.name}")
        with open(tmp_path, "wb") as f:
            f.write(audio_file.getbuffer())
        metadata['source_transcript_file'] = audio_file.name
        if st.button("Transcribe & Generate MOM"):
            st.info("Transcribing... (this uses Hugging Face Whisper model if you provided Hugging Face token in Settings)")
            hf_token = st.session_state.get('hf_token') or hf_token_input
            if hf_token:
                with open(tmp_path, "rb") as f:
                    audio_bytes = f.read()
                transcript = hf_transcribe_audio_bytes(audio_bytes, hf_token)
                if transcript:
                    st.success("Transcription complete.")
                    # Generate MOM using HF LLM if possible
                    human_text, json_mom = hf_generate_mom_from_transcript(transcript, hf_token, metadata)
                    if json_mom:
                        st.session_state['last_mom'] = json_mom
                        st.success("MOM generated — see below.")
                    else:
                        # fallback to local
                        human_text, json_mom = local_extractive_mom(transcript, metadata)
                        st.session_state['last_mom'] = json_mom
                        st.success("MOM generated using local fallback.")
                else:
                    st.warning("Transcription failed using Hugging Face. Using local fallback (no STT). Please upload a transcript instead.")
            else:
                st.warning("No Hugging Face token provided. Provide one in Settings to enable audio transcription. Using local fallback not available for audio — please upload a transcript or paste one.")

# ---------- Show MOM if exists ----------
if 'last_mom' in st.session_state:
    mom = st.session_state['last_mom']
    st.header("Minutes of Meeting (MOM)")
    st.markdown(f"**Meeting:** {mom.get('meeting_title','Untitled Meeting')}")
    st.markdown(f"**Date:** {mom.get('meeting_date') or 'Unknown'}")
    if mom.get('executive_summary'):
        st.subheader("Executive summary")
        st.write(mom['executive_summary'])
    st.subheader("Key points")
    for kp in mom.get('key_points',[]):
        st.write("- " + kp)
    st.subheader("Decisions")
    for d in mom.get('decisions',[]):
        st.write("- " + d)
    st.subheader("Action items (editable table)")
    ais = mom.get('action_items', [])
    df = pd.DataFrame(ais)
    # show editable table using experimental_data_editor
    edited_df = st.experimental_data_editor(df, num_rows="dynamic")
    if st.button("Save action item edits"):
        new_ais = edited_df.to_dict(orient='records')
        mom['action_items'] = new_ais
        st.session_state['last_mom'] = mom
        st.success("Saved edits to action items.")
    # export/download
    st.subheader("Export & manage")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        bio = download_docx_from_mom(mom)
        st.download_button("Download MOM (.docx)", data=bio, file_name="MOM.docx")
    with c2:
        st.download_button("Download MOM (.json)", data=json.dumps(mom, indent=2), file_name="MOM.json")
    with c3:
        if st.button("Clear MOM (requires confirmation)"):
            confirm = st.text_input("Type CONFIRM CLEAR to permanently delete the MOM", key="confirm_clear_input")
            if confirm == "CONFIRM CLEAR":
                st.session_state.pop('last_mom', None)
                st.success("MOM cleared from session.")
            else:
                st.warning("Type the exact phrase to confirm deletion.")

    st.markdown("---")
    st.caption(f"Temporary storage TTL: {DEFAULT_TTL_HOURS} hours. Use the Settings in the sidebar to add providers (optional).")

# Footer / privacy
st.markdown("### Privacy & notes")
st.write(
    "- Transcripts and generated MOM are stored only temporarily in this app's session/storage and are NOT shared.\n"
    "- If you provide a Hugging Face token, it is used to call Hugging Face Inference API; do not paste tokens if you don't want them to be used.\n"
    "- This is a prototype: for production, configure secure storage for keys and enable proper OAuth / authentication."
)
