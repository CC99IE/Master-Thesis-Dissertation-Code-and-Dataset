# modules/metadata.py

import os
import json
import base64
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from .common_warnings import *    # this sets up filters for PyPDF2
from .pil_config import *    # disables DecompressionBombWarning
from PIL import Image, ExifTags
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadWarning
from PyPDF2.generic import (
    BooleanObject, NumberObject, NameObject, TextStringObject,
    DictionaryObject, ArrayObject
)
import docx
import email
import email.parser

from PIL import Image, ExifTags
Image.MAX_IMAGE_PIXELS = None

# -------------------- CONFIGURATION --------------------

# Files to skip
IGNORED_FILES = {"Thumbs.db", ".DS_Store"}

# How many threads to spin up. Fallback to 4 if cpu_count() is None.
MAX_WORKERS = os.cpu_count() or 4

warnings.filterwarnings("ignore", category=PdfReadWarning)

# -------------------------------------------------------

def ensure_dir(path: Path):
    """
    Ensure that a directory exists.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def _serialize_val(v):
    """
    Recursively convert PyPDF2 generic objects (BooleanObject, NumberObject,
    NameObject, TextStringObject, DictionaryObject, ArrayObject) and raw bytes
    into pure Python types so that `json.dump()` will never fail.

    - PyPDF2 BooleanObject → bool
    - PyPDF2 NumberObject → int or float
    - PyPDF2 NameObject/TextStringObject → str
    - PyPDF2 DictionaryObject → dict
    - PyPDF2 ArrayObject → list
    - raw bytes         → Base64-encoded str
    - other types       → str(v) or None
    """
    # 1. Handle PyPDF2 generic types
    if isinstance(v, BooleanObject):
        return bool(v)

    if isinstance(v, NumberObject):
        # Some NumberObject are floats; if integer‐valued, cast to int
        f = float(v)
        return int(f) if f.is_integer() else f

    if isinstance(v, (NameObject, TextStringObject)):
        return str(v)

    if isinstance(v, DictionaryObject):
        result = {}
        for k, vv in v.items():
            # Keys in a DictionaryObject may themselves be NameObject/TextStringObject
            key_str = _serialize_val(k)
            result[key_str] = _serialize_val(vv)
        return result

    if isinstance(v, ArrayObject):
        return [_serialize_val(x) for x in v]

    # 2. Handle plain dict or list (just in case)
    if isinstance(v, dict):
        return { _serialize_val(k): _serialize_val(vv) for k, vv in v.items() }

    if isinstance(v, list):
        return [_serialize_val(x) for x in v]

    # 3. Handle raw bytes → Base64 string
    if isinstance(v, bytes):
        try:
            return base64.b64encode(v).decode('ascii')
        except Exception:
            return None

    # 4. Primitive Python types
    if isinstance(v, (bool, int, float, str)):
        return v

    # 5. Fallback to str(v)
    try:
        return str(v)
    except Exception:
        return None


def extract_pdf_metadata(path: Path) -> dict:
    """
    Extract metadata from a PDF using PyPDF2, converting any PyPDF2 generic
    objects into native Python types. Keys have leading '/' stripped.
    """
    meta = {}
    try:
        reader = PdfReader(str(path))
        doc_info = reader.metadata or {}
        for raw_key, raw_val in doc_info.items():
            clean_key = raw_key.lstrip('/')  # e.g. "/Author" → "Author"
            meta[clean_key] = _serialize_val(raw_val)
    except Exception:
        # If something goes wrong parsing the PDF, return whatever we have so far
        pass
    return meta


def extract_image_metadata(path: Path) -> dict:
    """
    Extract EXIF metadata and any other info from an image. EXIF tags are
    prefixed with "EXIF:" in the resulting dict. Other entries from img.info
    are also included (e.g. “dpi”, “compression”, etc.), subject to Base64
    encoding if necessary.
    """
    meta = {}
    try:
        with Image.open(str(path)) as img:
            # 1) EXIF data
            exif_dict = img._getexif() or {}
            for tag, val in exif_dict.items():
                name = ExifTags.TAGS.get(tag, tag)
                # Prefix EXIF keys so they stand out
                meta[f"EXIF:{name}"] = _serialize_val(val)

            # 2) Any other info fields (e.g., "dpi", "compression")
            for k, v in img.info.items():
                # Only set if EXIF didn’t already populate that key
                if k not in meta:
                    meta[k] = _serialize_val(v)
    except Exception:
        # If image is unreadable/corrupt, return empty dict
        pass
    return meta


def extract_docx_metadata(path: Path) -> dict:
    """
    Extract core_properties from a .docx file using python-docx.
    Only common fields are pulled; missing fields are skipped.
    """
    meta = {}
    try:
        doc = docx.Document(str(path))
        props = doc.core_properties
        for attr in (
            'author', 'title', 'subject',
            'last_modified_by', 'created', 'modified',
            'category', 'comments', 'identifier'
        ):
            val = getattr(props, attr, None)
            if val:
                # created/modified may be datetime; str() serializes them
                meta[attr] = str(val)
    except Exception:
        pass
    return meta


def extract_email_metadata(path: Path) -> dict:
    """
    Parse an .eml file's headers (From, To, CC, Subject, Date, Message-ID).
    """
    meta = {}
    try:
        with open(str(path), 'r', encoding='utf-8', errors='ignore') as f:
            msg = email.parser.Parser().parse(f)
        for hdr in ('From', 'To', 'CC', 'Subject', 'Date', 'Message-ID'):
            if msg[hdr]:
                meta[hdr] = msg[hdr]
    except Exception:
        pass
    return meta


def _worker(args):
    full_path, base = args
    fname = full_path.name
    if fname in IGNORED_FILES:
        return None

    rel = str(full_path.relative_to(base))
    rec = {'relative_path': rel, 'metadata': {}}

    try:
        ext = full_path.suffix.lower()
        if ext == '.pdf':
            rec['metadata'] = extract_pdf_metadata(full_path)
        elif ext in ('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif'):
            rec['metadata'] = extract_image_metadata(full_path)
        elif ext == '.docx':
            rec['metadata'] = extract_docx_metadata(full_path)
        elif ext in ('.eml',):
            rec['metadata'] = extract_email_metadata(full_path)
        else:
            # No extractor for this extension; leave metadata empty
            rec['metadata'] = {}
    except Exception as e:
        rec['error'] = str(e)

    return rec


def run_metadata_extraction(input_folder: str, output_folder: str) -> dict:
    inp = Path(input_folder)
    out = Path(output_folder)
    ensure_dir(out)

    # 1) Build a list of all files under input_folder
    tasks = []
    for root, _, files in os.walk(inp):
        for fn in files:
            full = Path(root) / fn
            tasks.append((full, inp))

    records = []
    counts = {
        'total': 0,
        'pdf': 0,
        'image': 0,
        'docx': 0,
        'email': 0,
        'other': 0,
        'errors': 0
    }

    # 2) Spin up a ThreadPool to process them concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(_worker, t): t for t in tasks}
        for fut in as_completed(future_to_task):
            rec = fut.result()
            if rec is None:
                # This file was in IGNORED_FILES
                continue

            records.append(rec)
            counts['total'] += 1

            # Track errors
            if 'error' in rec:
                counts['errors'] += 1

            # Increment file‐type buckets
            ext = Path(rec['relative_path']).suffix.lower()
            if ext == '.pdf':
                counts['pdf'] += 1
            elif ext in ('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif'):
                counts['image'] += 1
            elif ext == '.docx':
                counts['docx'] += 1
            elif ext in ('.eml',):
                counts['email'] += 1
            else:
                counts['other'] += 1

    # 3) Write detailed per-file JSON
    details_path = out / 'metadata_records.json'
    with details_path.open('w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # 4) Write summary JSON
    summary = {
        'counts': counts,
        'details_path': str(details_path)
    }
    summary_path = out / 'metadata_summary.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
