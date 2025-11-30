import streamlit as st
import openai
import faiss
import pickle
import numpy as np
import re
import json
import requests
import jwt  # from PyJWT
import streamlit.components.v1 as components
from markdown_it import MarkdownIt
import os

from index_builder import (
    sync_drive_and_rebuild_index_if_needed,
    INDEX_FILE,
    METADATA_FILE,
    STATE_FILE,
    list_drive_files,  # ✅ NEW: for debug visibility
)


# ----------------------------
# Google OAuth login gate
# ----------------------------
def google_login():
    """
    Require the user to sign in with a Google account and restrict access
    to @richmondchambers.com email addresses.
    """
    if "user_email" in st.session_state:
        return st.session_state["user_email"]

    params = st.experimental_get_query_params()
    if "code" in params:
        code = params["code"][0]

        token_response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": st.secrets["GOOGLE_CLIENT_ID"],
                "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
                "redirect_uri": st.secrets["GOOGLE_REDIRECT_URI"],
                "grant_type": "authorization_code",
            },
        )

        if token_response.status_code != 200:
            st.error(
                "Authentication with Google failed. Please refresh the page and try again."
            )
            st.stop()

        token_data = token_response.json()
        id_token = token_data.get("id_token")
        if not id_token:
            st.error("No ID token received from Google. Access cannot be granted.")
            st.stop()

        try:
            claims = jwt.decode(id_token, options={"verify_signature": False})
        except Exception:
            st.error("Could not decode ID token. Access cannot be granted.")
            st.stop()

        email = claims.get("email", "")
        hosted_domain = claims.get("hd", "")

        if email.endswith("@richmondchambers.com") or hosted_domain == "richmondchambers.com":
            st.session_state["user_email"] = email
            return email
        else:
            st.error("Access is restricted to employees of Richmond Chambers.")
            st.stop()

    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        "?response_type=code"
        f"&client_id={st.secrets['GOOGLE_CLIENT_ID']}"
        f"&redirect_uri={st.secrets['GOOGLE_REDIRECT_URI']}"
        "&scope=openid%20email"
        "&prompt=select_account"
        "&access_type=offline"
    )

    st.markdown("### Richmond Chambers – Internal Tool")
    st.write(
        "Please sign in with a Richmond Chambers Google Workspace account to access this app."
    )
    st.markdown(f"[Sign in with Google]({auth_url})")
    st.stop()


# ----------------------------
# OpenAI key + login enforcement
# ----------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]
user_email = google_login()


# ----------------------------
# Helpers
# ----------------------------
def format_for_email(response_text: str) -> str:
    formatted = response_text.replace("**", "")
    formatted = formatted.replace("\n\n", "\n")
    return formatted.strip()


def extract_text_from_uploaded_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception:
            return ""

    if name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""

    try:
        return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_pdf_text(uploaded_file) -> str:
    text = extract_text_from_uploaded_file(uploaded_file)
    if not text or len(text.strip()) < 50:
        st.warning(
            f"Could not extract much text from {uploaded_file.name}. "
            "If this PDF is scanned, export from Gemini as text or DOCX."
        )
    return text


def first_name_only(name: str) -> str:
    if not name:
        return "[Client]"
    name = name.strip()

    if name.startswith("[") and name.endswith("]"):
        return name

    parts = name.split()
    return parts[0] if parts else "[Client]"


def extract_name_from_filename(filename: str) -> str:
    if not filename:
        return "[Client]"

    base = re.sub(r"\.[^.]+$", "", filename)
    base = re.sub(r"[_\-]+", " ", base)

    noise = [
        "gemini", "transcript", "summary", "consultation", "call", "meeting",
        "recording", "notes", "full", "final", "post con", "post-con",
        "richmond", "chambers",
    ]
    pattern = r"\b(" + "|".join(map(re.escape, noise)) + r")\b"
    cleaned = re.sub(pattern, " ", base, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", cleaned)
    if m:
        return first_name_only(m.group(1).strip())

    return "[Client]"


def extract_prospect_name(enquiry: str) -> str:
    closings = ["regards,", "best,", "sincerely,", "thanks,", "kind regards,"]
    for closing in closings:
        match = re.search(
            closing + r"\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)",
            enquiry,
            re.IGNORECASE,
        )
        if match:
            return first_name_only(match.group(1))

    match = re.search(
        r"my name is\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", enquiry, re.IGNORECASE
    )
    if match:
        return first_name_only(match.group(1))

    return "[Client]"


def clean_transcript(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"Meeting started.*?\n", "", text, flags=re.IGNORECASE)
    return text.strip()


def chunk_text(text: str, max_words: int = 1800):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i : i + max_words])


def get_embedding(text: str, model: str = "text-embedding-3-small"):
    result = openai.embeddings.create(input=[text], model=model)
    return result.data[0].embedding


def call_llm(prompt: str, model: str = "gpt-5.1", temperature: float = 0.2) -> str:
    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


# ----------------------------
# Load FAISS Index and Metadata
# ----------------------------
@st.cache_resource(ttl=86400)
def load_index_and_metadata():
    """
    Daily Drive sync (TTL=1 day), then load index + metadata.
    """
    did_rebuild = sync_drive_and_rebuild_index_if_needed()

    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata, did_rebuild


def load_last_rebuilt_timestamp() -> str:
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            return state.get("last_rebuilt", "Unknown")
        return "Unknown"
    except Exception:
        return "Unknown"


index, metadata, did_rebuild = load_index_and_metadata()
last_rebuilt = load_last_rebuilt_timestamp()

if did_rebuild:
    st.toast("Immigration law knowledge refreshed from Drive.")


# ✅ NEW: Manual rebuild button (bypasses TTL)
col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("Force rebuild now"):
        with st.spinner("Forcing Drive sync + rebuild..."):
            forced = sync_drive_and_rebuild_index_if_needed()
        st.cache_resource.clear()
        st.success("Rebuild complete. Please refresh the page.")
with col_b:
    st.caption("Use this if you’ve added new knowledge and want an immediate refresh.")


# ✅ NEW: Debug expander to confirm Drive visibility
with st.expander("Debug: Drive knowledge status", expanded=False):
    st.write("STATE_FILE path:", STATE_FILE)
    st.write("STATE_FILE exists?:", os.path.exists(STATE_FILE))

    try:
        files_dbg = list_drive_files()
        st.write("Drive files visible to service account:", len(files_dbg))
        if len(files_dbg) == 0:
            st.warning(
                "0 files found. On Streamlit Cloud this usually means the folder is "
                "on a Shared Drive and list calls need supportsAllDrives/includeItemsFromAllDrives "
                "or the service account lacks access."
            )
    except Exception as e:
        st.error(f"Drive listing error: {e}")


def search_index(query: str, k: int = 5):
    query_embedding = get_embedding(query)
    distances, indices = index.search(
        np.array([query_embedding], dtype=np.float32),
        k,
    )
    results = []
    for i in indices[0]:
        if i < len(metadata):
            results.append(metadata[i])
    return results


# ----------------------------
# Prompt builders
# ----------------------------
def build_chunk_prompt(chunk: str) -> str:
    return f"""
You are an experienced UK immigration barrister creating internal notes from ONE chunk of a full Gemini transcript.
Work ONLY from this chunk. Do not invent facts. If something is unclear, say so.

Return notes in this structure:

A. Facts and Timeline (chunk-based)
- bullet points of concrete facts with dates, history, documents, goals.
- mark unclear facts with (unclear).

B. Client Objectives / Questions Raised
- bullet points.

C. Routes / Options Discussed (route-by-route notes)
- For each route/option mentioned in this chunk, include:
- Route/Option:
- Why relevant to client:
- Key requirements mentioned:
- Application to facts:
- Evidence position:
- Risks / suitability / discretion:
- Strategic choices:
- Preliminary conclusion stated:

D. Actions / Next Steps Mentioned
- bullet points marked Confirmed or Suggested.

E. Ambiguities / Missing Info Identified
- bullet points.

Transcript chunk:
\"\"\"{chunk}\"\"\"
""".strip()


def build_final_prompt(
    all_chunk_notes: str,
    gemini_summary: str,
    additional_instructions: str = "",
    client_name: str = "[Client]",
) -> str:
    return f"""
You are an experienced UK immigration barrister drafting a post-consultation follow-up email to a client.

IMPORTANT OPENING RULE:
- The email MUST begin with exactly: "Dear {client_name},"
- Put a blank line after the salutation.

You have:
1) Full transcript notes (controlling factual basis)
2) Gemini summary (supportive only)

Rules:
- Do NOT invent facts or advice.
- Do NOT add new routes beyond consultation.
- If ambiguous, flag what’s missing.
- If transcript conflicts with Gemini summary, follow transcript and note neutrally.
- Use second person ("you", "your").
- Your Instructions must be bullet points. Other sections prose.
- Where a route was discussed in any detail, devote at least one substantial paragraph.
- Opening pleasantry must be 2–3 sentences and end with a scene-setting lead-in.
- Example lead-in: "This email summarises the key instructions you gave, the advice we discussed, and the next steps including fees and timing."
- Do not use horizontal rules or markdown separators (no "---").
- Use normal email paragraph spacing. Avoid overly academic sectioning.
- Do not include headings for the opening or closing.
- **All numbered section headings (2–7) MUST be bold exactly as written below, using Markdown bold (e.g., **Heading**). Do not rename or omit them.**
- **Within section 3, add a few short bold subheadings to break up the prose. Use only bold (not H2/H3). Make them case-appropriate and limited in number (typically 2–5).**
  - Example subheadings you may use where relevant: **Key issues**, **Routes/options discussed**, **Evidence and risks**, **Recommendation**.
  - If multiple routes are discussed, you may use route-specific subheadings like **Skilled Worker route**, **Global Talent route**, etc.

OUTPUT STRUCTURE:
1. Opening Pleasantry (no heading)
- 2–3 sentences.
- Sentence 1: thanks / reference date.
- Sentence 2: empathy / reassurance.
- Sentence 3: lead-in explaining that the email summarises instructions, advice, fees and next steps.

2. **Your Instructions**
- Bullet point summary of KEY instructions and facts only (not every minor detail).
- Aim for ~8–15 bullets unless the case is unusually complex.
- Include: identity, immigration history, current status, intended route(s), timelines, dependants, critical constraints (e.g., absences, salary, documents, prior refusals).
- Exclude: repeated preferences, side questions already covered in advice, or minor narrative detail.
- If two bullets would say the same thing, keep the clearer one and drop the other.
- Last bullet MUST start "You are seeking advice on..."

3. **Summary of Discussion and Legal Advice**
Prose only, but broken up with a few short bold subheadings. Must include:
- issue framing
- route-by-route paragraphs covering: relevance, requirements, application to facts, evidence, risks, strategy, preliminary conclusion
- brief comparison if multiple routes
- overall recommendation + caveats

4. **Proposed Further Work**
Prose. Distinguish confirmed vs suggested. End with bullets of scope.

5. **Our Professional Fees**
Prose soft phrasing. Fixed-fee statement. Breakdowns. Checking service bullets. Instalments.

6. **Quote Validity**
Prose. 30 days validity, target date, caveat for scope changes.

7. **Next Steps**
Prose. Invite confirmation. Admin team engagement letter. Warm sign-off.

Additional instructions:
\"\"\"{additional_instructions.strip()}\"\"\"

Gemini summary (supportive only):
\"\"\"{gemini_summary.strip()}\"\"\"

Transcript notes (controlling):
\"\"\"{all_chunk_notes.strip()}\"\"\"
""".strip()


def build_claim_extraction_prompt(final_summary: str) -> str:
    return f"""
Extract every legal proposition from this draft summary.
Return ONLY valid JSON in this format:
[
  {{ "claim": "...", "topic": "short label" }},
  ...
]

Draft summary:
\"\"\"{final_summary}\"\"\"
""".strip()


def build_verification_pro
