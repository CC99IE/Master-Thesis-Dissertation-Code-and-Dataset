import os
import json
import filetype
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from .common_warnings import *    # this sets up filters for PyPDF2
from .pil_config import *    # disables DecompressionBombWarning

from PIL import Image, ExifTags
Image.MAX_IMAGE_PIXELS = None
# Files to ignore during ingestion
IGNORED_FILES = {"Thumbs.db", ".DS_Store"}

def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True)

def _process_file(args):
    """
    Worker for a single file: gathers stats, guesses mime, checks extension match.
    Returns (record_dict, is_mismatch:bool).
    """
    full_path, input_folder = args
    fname = full_path.name
    rel_path = str(full_path.relative_to(input_folder))
    # Skip ignored files
    if fname in IGNORED_FILES:
        return None, False

    # Gather filesystem metadata
    try:
        stats = full_path.stat()
    except OSError:
        return None, False

    size = stats.st_size
    atime, mtime, ctime = stats.st_atime, stats.st_mtime, stats.st_ctime

    # Declared extension
    ext = full_path.suffix.lower().lstrip('.') or ''

    # Detect MIME
    try:
        kind = filetype.guess(str(full_path))
        detected_mime = kind.mime if kind else 'unknown'
    except Exception:
        detected_mime = 'unknown'

    # Check if extension appears in the mime string
    ext_match = ext and (ext in detected_mime.lower())

    record = {
        'relative_path': rel_path,
        'size_bytes': size,
        'atime': atime,
        'mtime': mtime,
        'ctime': ctime,
        'declared_extension': ext,
        'detected_mime': detected_mime,
        'extension_matches_mime': bool(ext_match),
    }
    return record, not ext_match

def run_ingestion(input_folder: str, output_folder: str) -> dict:
    """
    Walks the input_folder, gathers file metadata and flags extension↔mime mismatches in parallel.
    Writes both detailed records and a summary into output_folder.
    """
    inp = Path(input_folder)
    out = Path(output_folder)
    ensure_dir(out)

    # Collect all candidate files
    all_paths = []
    for root, _, files in os.walk(inp):
        for fn in files:
            p = Path(root) / fn
            all_paths.append((p, inp))

    file_records = []
    mismatches = []

    # Process in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = { exe.submit(_process_file, args): args for args in all_paths }
        for future in as_completed(futures):
            rec, is_bad = future.result()
            if rec:
                file_records.append(rec)
                if is_bad:
                    mismatches.append(rec['relative_path'])

    # Write detailed records
    records_path = out / 'ingested_files.json'
    with records_path.open('w', encoding='utf-8') as f:
        json.dump(file_records, f, indent=2, ensure_ascii=False)

    # Write summary
    summary = {
        'total_files': len(file_records),
        'mismatched_extension_count': len(mismatches),
        'mismatched_files': mismatches,
        'records_path': str(records_path),
    }
    summary_path = out / 'ingestion_summary.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
