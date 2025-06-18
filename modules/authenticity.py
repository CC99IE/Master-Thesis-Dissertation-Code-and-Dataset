import os
import json
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from .common_warnings import *    # this sets up filters for PyPDF2
from .pil_config import *    # disables DecompressionBombWarning

# --- CONFIGURATION -----------------------------------------------------------

# How many bytes to sample per file for entropy
SAMPLE_SIZE = 1024 * 1024  # 1 MB

# Files to skip
IGNORED_FILES = {"Thumbs.db", ".DS_Store"}

# Entropy thresholds (bits per byte)
LOW_ENTROPY_THRESHOLD = 3.0   # too uniform
HIGH_ENTROPY_THRESHOLD = 7.5  # too random

# -----------------------------------------------------------------------------


def ensure_dir(path: Path):
    """Create `path` (and parents) if it doesn’t already exist."""
    if not path.exists():
        path.mkdir(parents=True)


def calculate_entropy(data: bytes) -> float:
    """
    Compute Shannon entropy (bits/byte) for the given data.
    Empty data => entropy 0.
    """
    length = len(data)
    if length == 0:
        return 0.0

    # byte-frequency histogram
    freq = [0] * 256
    for b in data:
        freq[b] += 1

    entropy = 0.0
    for count in freq:
        if count:
            p = count / length
            entropy -= p * math.log2(p)
    return entropy


def _entropy_worker(args):
    """
    Worker fn for a single file.
    Returns (record_dict, is_anomaly:bool) or (None, False) on skip/error.
    """
    full_path, base_folder = args
    fname = full_path.name
    if fname in IGNORED_FILES:
        return None, False

    rel_path = str(full_path.relative_to(base_folder))
    try:
        with full_path.open('rb') as f:
            data = f.read(SAMPLE_SIZE)
    except Exception:
        return None, False

    ent = calculate_entropy(data)
    record = {
        'relative_path': rel_path,
        'entropy': ent,
    }
    is_anomaly = (ent < LOW_ENTROPY_THRESHOLD) or (ent > HIGH_ENTROPY_THRESHOLD)
    return record, is_anomaly


def run_authenticity_checks(input_folder: str, output_folder: str) -> dict:
    """
    Walks `input_folder`, computes per-file entropy in parallel, flags anomalies,
    and writes both detailed records and a summary into `output_folder`.
    """
    inp = Path(input_folder)
    out = Path(output_folder)
    ensure_dir(out)

    # Gather all file paths
    tasks = []
    for root, _, files in os.walk(inp):
        for fn in files:
            tasks.append((Path(root) / fn, inp))

    entropy_records = []
    anomalies = []

    # Parallel entropy calculation
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = {exe.submit(_entropy_worker, args): args for args in tasks}
        for future in as_completed(futures):
            rec, is_anom = future.result()
            if not rec:
                continue
            entropy_records.append(rec)
            if is_anom:
                anomalies.append(rec)

    # Write detailed records
    details_path = out / 'authenticity_entropy_records.json'
    with details_path.open('w', encoding='utf-8') as f:
        json.dump(entropy_records, f, indent=2, ensure_ascii=False)

    # Build summary
    total = sum(r['entropy'] for r in entropy_records)
    count = len(entropy_records)
    summary = {
        'total_files_analyzed': count,
        'average_entropy': (total / count) if count else 0.0,
        'anomaly_count': len(anomalies),
        'anomalies': anomalies,
        'details_path': str(details_path),
    }

    summary_path = out / 'authenticity_summary.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
