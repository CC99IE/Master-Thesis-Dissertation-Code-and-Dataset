import os
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from .common_warnings import *    # this sets up filters for PyPDF2
from .pil_config import *    # disables DecompressionBombWarning
import numpy as np
from PIL import Image, ExifTags
import imagehash
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.content_analysis import extract_text


Image.MAX_IMAGE_PIXELS = None

# --- CONFIG ---

IGNORED_FILES = {"Thumbs.db", ".DS_Store"}
IMAGE_EXTS      = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif'}
TEXT_EXTS       = {'.pdf', '.docx', '.txt', '.eml', '.msg'}

IMAGE_HASH_THRESHOLD   = 5      # max Hamming distance (bits)
TEXT_SIM_THRESHOLD      = 0.8    # cosine similarity cutoff

MAX_WORKERS = os.cpu_count() or 4

# --- HELPERS ---

def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def _hash_image(task):
    """
    Worker for ProcessPool: returns (rel_path, hex_hash, bit_list) or None on failure.
    """
    full_path, rel_path = task
    try:
        with Image.open(full_path) as img:
            ph = imagehash.phash(img)
        bits = ph.hash.flatten().astype(np.uint8).tolist()
        return rel_path, str(ph), bits
    except Exception:
        return None


def _extract_text(task):
    """
    Worker for ProcessPool: returns (rel_path, text) or None on failure.
    """
    full_path, rel_path, ext = task
    try:
        text = extract_text(str(full_path), ext)
        return rel_path, text
    except Exception:
        return None

# --- MAIN ENTRY ---

def run_similarity_scoring(input_folder: str, output_folder: str) -> dict:
    """
    1) Parallel image‐hashing → radius‐search Hamming
    2) Parallel text‐extraction → TF–IDF cosine sim
    3) Simple mtime “burstiness” timeline

    Writes:
      - image_similarity.json
      - text_similarity.json
      - timeline.json
      - similarity_summary.json
    """
    inp = Path(input_folder)
    out = Path(output_folder)
    ensure_dir(out)

    # 1) walk once, classify tasks + gather mtimes
    image_tasks = []
    text_tasks  = []
    timeline    = []
    total_files = 0

    for root, _, files in os.walk(inp):
        root = Path(root)
        for fn in files:
            if fn in IGNORED_FILES:
                continue
            total_files += 1
            p    = root / fn
            rel  = str(p.relative_to(inp))
            ext  = p.suffix.lower()

            # record for timeline
            try:
                timeline.append({'relative_path': rel, 'mtime': p.stat().st_mtime})
            except OSError:
                pass

            # enqueue
            if ext in IMAGE_EXTS:
                image_tasks.append((p, rel))
            elif ext in TEXT_EXTS:
                text_tasks.append((p, rel, ext))

    # 2) hash images in parallel
    image_hashes = {}
    bit_rows     = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(_hash_image, t): t for t in image_tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if not res:
                continue
            rel_path, hex_hash, bits = res
            image_hashes[rel_path] = hex_hash
            bit_rows.append((rel_path, np.array(bits, dtype=np.uint8)))

    # 3) build neighbor‐search for images (if any)
    image_similar = []
    if bit_rows:
        keys = [r for r,_ in bit_rows]
        arr  = np.stack([row for _, row in bit_rows], axis=0)
        BIT_COUNT = arr.shape[1]
        radius = IMAGE_HASH_THRESHOLD / BIT_COUNT

        nbr = NearestNeighbors(radius=radius, metric='hamming', n_jobs=-1)
        nbr.fit(arr)
        dists, neighs = nbr.radius_neighbors(arr, return_distance=True)

        for i, nbr_idxs in enumerate(neighs):
            for d_frac, j in zip(dists[i], nbr_idxs):
                if j <= i:
                    continue
                hd = int(round(d_frac * BIT_COUNT))
                image_similar.append({
                    'file1': keys[i],
                    'file2': keys[j],
                    'hamming_distance': hd
                })

    # 4) extract text in parallel
    text_paths = []
    text_docs  = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(_extract_text, t): t for t in text_tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if not res:
                continue
            rel_path, txt = res
            if txt and txt.strip():
                text_paths.append(rel_path)
                text_docs.append(txt)

    # 5) cosine‐similarity on TF–IDF
    text_similar = []
    if text_docs:
        vec   = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf = vec.fit_transform(text_docs)
        simm  = cosine_similarity(tfidf)
        n = len(text_paths)
        for i in range(n):
            for j in range(i+1, n):
                score = float(simm[i, j])
                if score >= TEXT_SIM_THRESHOLD:
                    text_similar.append({
                        'file1': text_paths[i],
                        'file2': text_paths[j],
                        'cosine_similarity': score
                    })

    # 6) timeline: group by day
    day_counts = {}
    for e in timeline:
        day = datetime.fromtimestamp(e['mtime']).date().isoformat()
        day_counts[day] = day_counts.get(day, 0) + 1

    # --- WRITE OUTPUTS ---
    def _dump(obj, fname):
        p = out / fname
        with p.open('w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2)
        return str(p)

    images_path   = _dump(image_similar,      'image_similarity.json')
    texts_path    = _dump(text_similar,       'text_similarity.json')
    timeline_path = _dump({'day_counts': day_counts, 'raw': timeline}, 'timeline.json')

    summary = {
        'total_files': total_files,
        'images_hashed': len(image_hashes),
        'image_pairs_flagged': len(image_similar),
        'texts_extracted': len(text_docs),
        'text_pairs_flagged': len(text_similar),
        'days_of_activity': len(day_counts),
        'image_similarity_path': images_path,
        'text_similarity_path': texts_path,
        'timeline_path': timeline_path
    }
    _dump(summary, 'similarity_summary.json')

    return summary
