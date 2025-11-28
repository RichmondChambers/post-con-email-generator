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

from index_builder import (
    sync_drive_and_rebuild_index_if_needed,
    INDEX_FILE,
    METADATA_FILE,
)


# ----------------------------
# Google OAuth login gate
# ----------------------------
def google_login():
    """
    Require the user to sign in with a Google account and restrict access
    to @richmondchambers.com email addresses.
    """
    # 1. If we already have a logged-in user in this session, allow access
    if "user_email" in st.session_state:
        return st.session_state["user_email"]

    # 2. Check if Google has redirected back with a ?code=... parameter
    params = st.experimental_get_query_params()
    if "code" in params:
        code = params["code"][0]

        # Exchange the code for tokens
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

        # Decode the ID token to get the user's email address.
        # For simplicity we skip signature verification here.
        # For a stricter setup, you would verify the token using Google's public keys.
        try:
            claims = jwt.decode(id_token, options={"verify_signature": False})
        except Exception:
            st.error("Could not decode ID token. Access cannot be granted.")
            st.stop()

        email = claims.get("email", "")
        hosted_domain = claims.get("hd", "")  # sometimes set to 'richmondchambers.com'

        # Enforce @richmondchambers.com
        if email.endswith("@richmondchambers.com") or hosted_domain == "richmondchambers.com":
            st.session_state["user_email"] = email
            return email
        else:
            st.error("Access is restricted to employees of Richmond Chambers.")
            st.stop()

    # 3. If we get here, the user is not yet logged in.
    # Show a "Sign in with Google" link that starts the OAuth flow.
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

    # Stop the app here until the user has logged in
    st.stop()


# ----------------------------
# OpenAI key + login enforcement
# ----------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 🔐 Enforce Google sign-in for @richmondchambers.com
user_email = google_login()

# (Optional debug)
# st.write(f"Signed in as: {user_email}")


# ----------------------------
# Helpers
# ----------------------------
def format_for_email(response_text: str) -> str:
    """
    Cleans up the AI response so it's suitable for copying into an email.
    Removes Markdown and extra spacing.
    """
    formatted = response_text.replace("**", "")  # remove bold markup
    formatted = formatted.replace("\n\n", "\n")  # remove extra spacing
    return formatted.strip()


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """
    Extract text content from an uploaded file.
    Supports .txt directly; for PDF/DOCX you will need the relevant libraries
    installed (PyPDF2 / python-docx).
    """
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

    # Fallback – try to decode as text
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
    """Convert a full name like "John Smith" to "John". Leaves placeholders like "[Client]" untouched."""
    if not name:
        return "[Client]"
    name = name.strip()

    # Don't touch placeholders
    if name.startswith("[") and name.endswith("]"):
        return name

    parts = name.split()
    return parts[0] if parts else "[Client]"


def extract_name_from_filename(filename: str) -> str:
    """
    Try to infer a client first name (or full name) from the transcript PDF filename.
    Returns "[Client]" if nothing reliable is found.
    """
    if not filename:
        return "[Client]"

    # strip extension
    base = re.sub(r"\.[^.]+$", "", filename)

    # replace separators with spaces
    base = re.sub(r"[_\-]+", " ", base)

    # remove common noise words
    noise = [
        "gemini",
        "transcript",
        "summary",
        "consultation",
        "call",
        "meeting",
        "recording",
        "notes",
        "full",
        "final",
        "post con",
        "post-con",
        "richmond",
        "chambers",
    ]
    pattern = r"\b(" + "|".join(map(re.escape, noise)) + r")\b"
    cleaned = re.sub(pattern, " ", base, flags=re.IGNORECASE)

    # collapse whitespace
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    # Look for 1–3 capitalised words in a row (likely a name)
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
@st.cache_resource
def load_index_and_metadata():
    """
    Ensure FAISS index is up to date, then load index, metadata, and
    read last rebuilt timestamp for UI display.
    """
    sync_drive_and_rebuild_index_if_needed()

    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    # Read the timestamp from drive_index_state.json
    try:
        with open("drive_index_state.json", "r") as f:
            state = json.load(f)
        last_rebuilt = state.get("last_rebuilt", "Unknown")
    except Exception:
        last_rebuilt = "Unknown"

    return index, metadata, last_rebuilt


index, metadata, last_rebuilt = load_index_and_metadata()


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
For each route/option mentioned in this chunk, include:
Route/Option:
Why relevant to client:
Key requirements mentioned:
Application to facts:
Evidence position:
Risks / suitability / discretion:
Strategic choices:
Preliminary conclusion stated:

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
Prose only. Must include:
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


def build_verification_prompt(
    final_summary: str, claims_json: str, sources_text: str
) -> str:
    return f"""
You are verifying legal accuracy of a post-consultation summary against internal Richmond Chambers knowledge.

Rules:
- Use internal sources as authoritative.
- If a claim is not clearly supported, flag it.
- Do not invent new law.

Draft summary:
\"\"\"{final_summary}\"\"\"

Extracted claims:
\"\"\"{claims_json}\"\"\"

Internal sources:
\"\"\"{sources_text}\"\"\"

For each claim, return:
- claim
- supported_status: Supported / Partially supported / Not supported
- explanation (brief, neutral)
- what to revise (if needed)

Return in clear numbered prose (not JSON).
""".strip()


# ----------------------------
# Streamlit app UI
# ----------------------------
st.markdown(
    """
<div style="text-align: center; padding-bottom: 10px;">
  <img src="https://raw.githubusercontent.com/RichmondChambers/richmond-immigration-assistant/main/assets/logo.png" width="150">
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center; font-size: 2.6rem;'>Post-Con Email Generator</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='color: grey; text-align: center; font-size: 0.9rem;'>"
    f"Immigration law knowledge last rebuilt from Drive on: <b>{last_rebuilt}</b></p>",
    unsafe_allow_html=True,
)

st.markdown("Upload the full Gemini transcript PDF and the Gemini summary PDF.")

full_pdf = st.file_uploader("Full Gemini transcript (PDF)", type=["pdf"])
summary_pdf = st.file_uploader("Gemini summary (PDF)", type=["pdf"])

additional_instructions = st.text_area(
    "Additional instructions (optional)",
    height=150,
)

do_faiss_check = st.checkbox(
    "Run legal accuracy check against internal knowledge (optional)"
)

generate = st.button("Generate Post-Con Email")


if generate:
    if not full_pdf or not summary_pdf:
        st.error("Please upload BOTH the full transcript PDF and the Gemini summary PDF.")
        st.stop()

    # 1) Extract text from PDFs
    full_text = extract_pdf_text(full_pdf)
    summary_text = extract_pdf_text(summary_pdf)
    transcript = clean_transcript(full_text)

    # 0) Try filename first
    client_name = extract_name_from_filename(full_pdf.name)

    # 1) If filename didn't yield anything, try transcript
    if client_name == "[Client]":
        client_name = extract_prospect_name(transcript)

    # 2) If still not found, try summary
    if client_name == "[Client]":
        client_name = extract_prospect_name(summary_text)

    # Final fallback safety
    if not client_name:
        client_name = "[Client]"

    # 2) Stage A: chunk notes from full transcript
    with st.spinner("Reviewing transcript and drafting post-con email..."):
        chunk_notes = []
        for chunk in chunk_text(transcript):
            chunk_notes.append(call_llm(build_chunk_prompt(chunk), temperature=0.1))

        combined_notes = "\n\n".join(chunk_notes)

        # 3) Stage B: final standardized post-con email summary
        final_prompt = build_final_prompt(
            combined_notes,
            summary_text,
            additional_instructions,
            client_name=client_name,
        )
        final_summary = call_llm(final_prompt, temperature=0.2)

    # Safety net: prepend salutation if model didn't
    if not final_summary.lstrip().lower().startswith("dear "):
        final_summary = f"Dear {client_name},\n\n{final_summary.lstrip()}"

    st.success("Post-con email generated.")
    st.text_area("Email-ready summary", value=final_summary, height=650)

    # 4) OPTIONAL: FAISS legal accuracy check
    if do_faiss_check:
        with st.spinner("Running legal accuracy check..."):
            claims_prompt = build_claim_extraction_prompt(final_summary)
            claims_text = call_llm(claims_prompt, temperature=0.0)

            try:
                claims = json.loads(claims_text)
            except Exception:
                claims = []

            retrieved_sources = []
            for c in claims:
                q = c.get("claim") or c.get("topic")
                if not q:
                    continue
                retrieved_sources.extend(search_index(q, k=3))

            unique_sources = []
            seen = set()
            for s in retrieved_sources:
                content = s.get("content", "")
                if content and content not in seen:
                    unique_sources.append(s)
                    seen.add(content)

            # NOTE: no horizontal rule separators to avoid GDocs HR artifacts
            sources_text = "\n\n".join(
                f"[{s.get('source','internal')}]\n{s.get('content','')}"
                for s in unique_sources
            )

            verification_prompt = build_verification_prompt(
                final_summary, claims_text, sources_text
            )
            verification_report = call_llm(verification_prompt, temperature=0.0)

        with st.expander("Legal accuracy check (internal use only)", expanded=False):
            st.markdown(verification_report)

    # --- ALWAYS show responsibility statement ---
    st.markdown(
        """
**Professional Responsibility Statement**

AI-generated content must not be relied upon without human review. Where such content is used,
the barrister is responsible for verifying and ensuring the accuracy and legal soundness of that content.
AI tools are used solely to support drafting and research; they do not replace the barrister’s independent
judgment, analysis, or duty of care.
""",
        unsafe_allow_html=False,
    )

    # --- Copy to clipboard button (uses final_summary) ---
    md = MarkdownIt()
    html_reply = md.render(final_summary)

    # Escape for safe JS embedding
    html_js = json.dumps(html_reply)
    text_js = json.dumps(final_summary)

    components.html(
        f"""
<style>
.copy-button {{
  margin-top: 10px;
  padding: 8px 16px;
  background-color: #2e2e2e;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s ease, transform 0.1s ease;
}}
.copy-button:hover {{
  background-color: #4a4a4a;
}}
.copy-button:active {{
  background-color: #3a3a3a;
  transform: scale(0.98);
}}
</style>

<button class="copy-button" onclick="copyToClipboard()">📋 Copy to Clipboard</button>

<script>
async function copyToClipboard() {{
  const htmlContent = {html_js};
  const plainText = {text_js};

  const blobHtml = new Blob([htmlContent], {{ type: 'text/html' }});
  const blobText = new Blob([plainText], {{ type: 'text/plain' }});

  const clipboardItem = new ClipboardItem({{
    'text/html': blobHtml,
    'text/plain': blobText
  }});

  await navigator.clipboard.write([clipboardItem]);
  alert("Formatted text copied! Paste into Gmail or Google Docs to retain formatting.");
}}
</script>
""",
        height=120,
        scrolling=False,
    )
