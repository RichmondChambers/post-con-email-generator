import os
import io
import json
import pickle
import time
import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional

import numpy as np
import faiss
import openai
import docx
import PyPDF2

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import streamlit as st

# ----------------------------
# Paths (robust on Streamlit Cloud)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔹 Folder ID of your UK-Immigration-Knowledge folder in Drive
DRIVE_FOLDER_ID = "13J-DiERhtS1VWgF2GtZ1wnMfbUzkq6-G"

# 🔹 Local files the app uses (absolute paths)
INDEX_FILE = os.path.join(BASE_DIR, "faiss_index.index")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.pkl")
STATE_FILE = os.path.join(BASE_DIR, "drive_index_state.json")  # detect changes + timestamp

# 🔹 Drive sync behavior
OVERNIGHT_WINDOW_START = int(os.getenv("DRIVE_SYNC_WINDOW_START_HOUR", "1"))  # 01:00 UK
OVERNIGHT_WINDOW_END = int(os.getenv("DRIVE_SYNC_WINDOW_END_HOUR", "6"))    # 06:00 UK
COOLDOWN_HOURS = int(os.getenv("DRIVE_REBUILD_COOLDOWN_HOURS", "20"))
ALLOW_DAYTIME_DRIVE_CHECKS = (
    os.getenv("ENABLE_DAYTIME_DRIVE_CHECKS", "false").lower() == "true"
)


def get_drive_service():
    """
    Build an authenticated Google Drive API client using the service account
    stored in st.secrets["gcp_service_account"].
    """
    creds_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    service = build("drive", "v3", credentials=credentials)
    return service


def list_files_recursive(folder_id: str, service) -> List[Dict]:
    """
    Recursively list all non-folder files under a Drive folder (including sub-folders).
    Supports Shared Drives via supportsAllDrives/includeItemsFromAllDrives.
    """
    files: List[Dict] = []
    page_token = None

    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            pageToken=page_token,
            supportsAllDrives=True,          # ✅ critical for Shared Drives
            includeItemsFromAllDrives=True,  # ✅ critical for Shared Drives
        ).execute()

        for f in response.get("files", []):
            mime_type = f.get("mimeType", "")
            if mime_type == "application/vnd.google-apps.folder":
                files.extend(list_files_recursive(f["id"], service))
            else:
                files.append(f)

        page_token = response.get("nextPageToken")
        if page_token is None:
            break

    return files


def list_drive_files() -> List[Dict]:
    """
    Return a list of all files (id, name, mimeType, modifiedTime)
    under the main knowledge folder (including sub-folders).
    """
    service = get_drive_service()
    return list_files_recursive(DRIVE_FOLDER_ID, service)


def load_previous_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def have_files_changed(current_files, previous_state):
    """
    Compare current Drive files with previous state to see if anything is new or modified.
    """
    current_state = {f["id"]: f["modifiedTime"] for f in current_files}
    if current_state != previous_state.get("files", {}):
        return True, current_state
    return False, current_state


def _now_london(now_utc: Optional[datetime.datetime] = None) -> datetime.datetime:
    if now_utc is None:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
    return now_utc.astimezone(ZoneInfo("Europe/London"))


def _within_overnight_window(now_london: datetime.datetime) -> bool:
    start = datetime.time(hour=OVERNIGHT_WINDOW_START)
    end = datetime.time(hour=OVERNIGHT_WINDOW_END)
    if start <= end:
        return start <= now_london.time() < end
    # window wraps midnight
    return now_london.time() >= start or now_london.time() < end


def _parse_last_rebuilt(last_rebuilt: str) -> Optional[datetime.datetime]:
    if not last_rebuilt:
        return None
    if last_rebuilt.endswith("Z"):
        last_rebuilt = last_rebuilt[:-1]
    try:
        return datetime.datetime.fromisoformat(last_rebuilt).replace(tzinfo=datetime.timezone.utc)
    except Exception:
        return None


def download_file_bytes(service, file):
    """
    Download the raw bytes of a file from Google Drive.
    Handles both normal files (pdf/docx/txt) and Google Docs (exported as DOCX).
    """
    file_id = file["id"]
    mime_type = file.get("mimeType", "")

    if mime_type == "application/vnd.google-apps.document":
        request = service.files().export_media(
            fileId=file_id,
            mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        effective_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        request = service.files().get_media(
            fileId=file_id,
            supportsAllDrives=True  # ✅ allow downloads from Shared Drives too
        )
        effective_mime = mime_type

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read(), effective_mime


def extract_text_from_bytes(file_bytes: bytes, mime_type: str, file_name: str) -> str:
    """
    Convert downloaded bytes into plain text, depending on MIME type / extension.
    Supports DOCX, PDF, TXT, MD.
    """
    name_lower = file_name.lower()

    if (
        mime_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or name_lower.endswith(".docx")
    ):
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    if mime_type == "application/pdf" or name_lower.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)

    if mime_type.startswith("text/") or name_lower.endswith((".txt", ".md")):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="ignore")

    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def split_into_chunks(text: str, max_chars: int = 1500, overlap: int = 200):
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def embed_texts(texts, model="text-embedding-3-small", batch_size=16) -> np.ndarray:
    """
    Get embeddings for a list of texts using OpenAI embeddings API,
    with basic rate-limit handling.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        while True:
            try:
                response = openai.embeddings.create(input=batch, model=model)
                break
            except openai.RateLimitError as e:
                wait_seconds = 5
                print(
                    f"[index_builder] Rate limit hit, sleeping {wait_seconds}s and retrying batch {i // batch_size}: {e}"
                )
                time.sleep(wait_seconds)

        for item in response.data:
            all_embeddings.append(item.embedding)

    return np.array(all_embeddings, dtype=np.float32)


def rebuild_index_from_drive(files: List[Dict]):
    """
    Download files, extract text, chunk, embed, and rebuild FAISS + metadata.
    """
    service = get_drive_service()
    all_chunks = []
    metadata = []

    for file in files:
        file_id = file["id"]
        file_name = file.get("name", "unnamed")
        mime_type = file.get("mimeType", "")

        if mime_type == "application/vnd.google-apps.folder":
            continue

        file_bytes, effective_mime = download_file_bytes(service, file)
        text = extract_text_from_bytes(file_bytes, effective_mime, file_name)
        if not text.strip():
            continue

        chunks = split_into_chunks(text)

        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append(
                {
                    "content": chunk,
                    "file_id": file_id,
                    "file_name": file_name,
                    "chunk_index": idx,
                }
            )

    if not all_chunks:
        dim = 1536
        index = faiss.IndexFlatL2(dim)
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump([], f)
        return

    embeddings = embed_texts(all_chunks, model="text-embedding-3-small")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)


def sync_drive_and_rebuild_index_if_needed(
    bypass_cooldown: bool = False,
    allow_daytime_checks: Optional[bool] = None,
    now_utc: Optional[datetime.datetime] = None,
):
    """
    Check Drive for new/updated files; if changes are detected, rebuild FAISS index
    and metadata from scratch.

    Cooldown + overnight window behaviour:
    - The app only checks Drive during the UK overnight window by default so daytime
      users skip the expensive Drive sync.
    - A daily cooldown (configurable) prevents repeated Drive scans unless
      `bypass_cooldown` is True (used by the nightly job).
    - If local artifacts are missing, rebuild immediately regardless of cooldown
      or window.
    """

    allow_daytime = ALLOW_DAYTIME_DRIVE_CHECKS if allow_daytime_checks is None else allow_daytime_checks
    now_london = _now_london(now_utc)
    previous_state = load_previous_state()
    last_rebuilt = _parse_last_rebuilt(previous_state.get("last_rebuilt", ""))

    local_missing = not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE)
    if local_missing:
        files = list_drive_files()
        rebuild_index_from_drive(files)
        save_state(
            {
                "files": {f["id"]: f["modifiedTime"] for f in files},
                "last_rebuilt": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )
        return True

    if not allow_daytime and not _within_overnight_window(now_london):
        return False

    if not bypass_cooldown and last_rebuilt:
        elapsed = now_london - last_rebuilt.astimezone(ZoneInfo("Europe/London"))
        if elapsed < datetime.timedelta(hours=COOLDOWN_HOURS):
            return False

    files = list_drive_files()
    changed, current_state = have_files_changed(files, previous_state)

    if changed:
        rebuild_index_from_drive(files)
        save_state(
            {
                "files": current_state,
                "last_rebuilt": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )
        return True

    # ✅ If state file doesn't exist yet, write one anyway so banner isn't stuck on Unknown
    if not os.path.exists(STATE_FILE):
        save_state(
            {
                "files": current_state,
                "last_rebuilt": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )

    return False
