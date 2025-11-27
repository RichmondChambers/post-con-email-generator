import os
import io
import json
import pickle
import time
import datetime
from typing import List, Dict

import numpy as np
import faiss
import openai
import docx
import PyPDF2

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import streamlit as st

# 🔹 Folder ID of your UK-Immigration-Knowledge folder in Drive
# You already had this ID in your previous code:
DRIVE_FOLDER_ID = "13J-DiERhtS1VWgF2GtZ1wnMfbUzkq6-G"

# 🔹 Local files the app uses
INDEX_FILE = "faiss_index.index"
METADATA_FILE = "metadata.pkl"
STATE_FILE = "drive_index_state.json"  # to detect changes (optional)


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
    """
    files: List[Dict] = []

    # First, list direct children of this folder
    page_token = None
    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            pageToken=page_token,
        ).execute()

        for f in response.get("files", []):
            mime_type = f.get("mimeType", "")
            if mime_type == "application/vnd.google-apps.folder":
                # Recurse into sub-folder
                files.extend(list_files_recursive(f["id"], service))
            else:
                files.append(f)

        page_token = response.get("nextPageToken", None)
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
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
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


def download_file_bytes(service, file):
    """
    Download the raw bytes of a file from Google Drive.
    Handles both normal files (pdf/docx/txt) and Google Docs (exported as DOCX).
    """
    file_id = file["id"]
    mime_type = file.get("mimeType", "")

    if mime_type == "application/vnd.google-apps.document":
        # Google Docs → export as DOCX
        request = service.files().export_media(
            fileId=file_id,
            mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        effective_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        request = service.files().get_media(fileId=file_id)
        effective_mime = mime_type

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read(), effective_mime


def extract_text_from_bytes(file_bytes: bytes, mime_type: str, file_name: str) -> str:
    """
    Convert downloaded bytes into plain text, depending on MIME type / extension.
    Supports DOCX, PDF, TXT, MD. Extend if needed.
    """
    name_lower = file_name.lower()

    # DOCX (including exported Google Docs)
    if (
        mime_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or name_lower.endswith(".docx")
    ):
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    # PDF
    if mime_type == "application/pdf" or name_lower.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages)

    # Plain text
    if mime_type.startswith("text/") or name_lower.endswith(".txt") or name_lower.endswith(".md"):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="ignore")

    # Fallback
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def split_into_chunks(text: str, max_chars: int = 1500, overlap: int = 200):
    """
    Simple character-based chunking with overlap.
    """
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

    - Uses smaller batches (default 16) to reduce tokens per request.
    - If a RateLimitError is hit, waits a few seconds and retries.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        while True:
            try:
                response = openai.embeddings.create(
                    input=batch,
                    model=model,
                )
                break  # success, exit the retry loop
            except openai.RateLimitError as e:
                # Simple backoff: wait then retry the same batch
                wait_seconds = 5
                print(f"[index_builder] Rate limit hit, sleeping {wait_seconds}s and retrying batch {i // batch_size}: {e}")
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

        # Skip if it's a folder (shouldn't happen here, but just in case)
        if mime_type == "application/vnd.google-apps.folder":
            continue

        # 1. Download file bytes
        file_bytes, effective_mime = download_file_bytes(service, file)

        # 2. Extract text
        text = extract_text_from_bytes(file_bytes, effective_mime, file_name)
        if not text.strip():
            continue

        # 3. Split into chunks
        chunks = split_into_chunks(text)

        # 4. Add to master list with metadata
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
        # No text → create an empty index
        dim = 1536  # embedding dimension for text-embedding-3-small
        index = faiss.IndexFlatL2(dim)
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump([], f)
        return

    # 5. Embed chunks
    embeddings = embed_texts(all_chunks, model="text-embedding-3-small")

    # 6. Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 7. Save index and metadata
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
            pickle.dump(metadata, f)


def sync_drive_and_rebuild_index_if_needed():
    """
    Check Drive for new/updated files; if changes are detected,
    rebuild FAISS index and metadata from scratch.
    """
    files = list_drive_files()
    previous_state = load_previous_state()
    changed, current_state = have_files_changed(files, previous_state)

    if changed or not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        rebuild_index_from_drive(files)

        # Save new state + timestamp of rebuild
        save_state({
            "files": current_state,
            "last_rebuilt": datetime.datetime.utcnow().isoformat() + "Z"
        })
        return True

    return False

