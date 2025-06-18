# Main.py

import os
import sys
import argparse
import traceback
import json
import time
import warnings
import logging

# ──────────────────────────────────────────────────────────────────────────────
# 1) GLOBAL WARNING / LOGGER CONFIGURATION
#    – Suppress all PyPDF2 PdfReadWarning messages
#    – Force PyPDF2’s logger to only show ERROR+ messages
from PyPDF2.errors import PdfReadWarning
warnings.filterwarnings("ignore", category=PdfReadWarning)
logging.getLogger("PyPDF2").setLevel(logging.ERROR)

# 2) GLOBAL PIL CONFIGURATION
#    – Disable “DecompressionBombWarning” Spam
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
from modules.ingestion               import run_ingestion
from modules.authenticity            import run_authenticity_checks as run_authenticity
from modules.metadata                import run_metadata_extraction as run_metadata
from modules.content_analysis        import run_content_analysis
from modules.similarity              import run_similarity_scoring as run_similarity
from modules.ml_classifier           import run_ml_classifier
from modules.reporting               import run_reporting
# ──────────────────────────────────────────────────────────────────────────────
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def count_file_types(ingested_json_path: str) -> dict:
    counts = {}
    if not os.path.exists(ingested_json_path):
        return counts

    with open(ingested_json_path, 'r', encoding='utf-8') as f:
        try:
            records = json.load(f)
        except Exception:
            return counts

    for rec in records:
        rel = rec.get('relative_path') or rec.get('path') or ''
        ext = os.path.splitext(rel)[1].lower() or '[no ext]'
        counts[ext] = counts.get(ext, 0) + 1

    return counts


def print_summary(json_path: str, title: str):
    if not os.path.exists(json_path):
        print(f"[Warning] Missing summary: {title}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception:
            print(f"[Warning] Could not parse JSON summary: {title}")
            return

    print(f"\n--- {title} ---")
    if isinstance(data, dict):
        for k, v in data.items():
            field_name = k.replace('_', ' ').title()
            if isinstance(v, list):
                print(f"{field_name}: {len(v)} item(s)")
            elif isinstance(v, dict):
                print(f"{field_name}: {len(v)} field(s)")
            else:
                print(f"{field_name}: {v}")
    else:
        print(f"(Unexpected format—type {type(data).__name__}, length {len(data)})")


def generate_charts(input_dir: str, output_dir: str):
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # 1) Bar chart of module runtimes
    timings_json = os.path.join(output_dir, 'module_timings.json')
    if os.path.exists(timings_json):
        try:
            from modules.charts import plot_module_timings
            png1 = plot_module_timings(timings_json, charts_dir)
            print(f"[Charts] Module runtimes chart → {png1}")
        except Exception as e:
            print(f"[Charts] ERROR generating module runtimes chart: {e}")
    else:
        print("[Charts] WARNING: module_timings.json not found; skipping runtimes chart")

    # 2) Histogram of entropy
    entropy_json = os.path.join(output_dir, 'authenticity_entropy_records.json')
    if os.path.exists(entropy_json):
        try:
            from modules.charts import plot_entropy_histogram
            png2 = plot_entropy_histogram(entropy_json, charts_dir)
            print(f"[Charts] Entropy histogram → {png2}")
        except Exception as e:
            print(f"[Charts] ERROR generating entropy histogram: {e}")
    else:
        print("[Charts] WARNING: authenticity_entropy_records.json not found; skipping entropy histogram")


def main(input_dir: str, output_dir: str, contamination: float, random_state: int):
    """
    Orchestrates the entire pipeline:
      1. Ingestion & Sanity-Checks
      2. Authenticity & Anomaly Detection
      3. Metadata Extraction
      4. Content Analysis & NLP
      5. Similarity & Correlation Scoring
      6. ML Fake-vs-Real Classifier
      7. Generate Charts
      8. Reporting & Export
    """
    ensure_dir(output_dir)
    overall_start = time.perf_counter()

    # Define the pipeline steps (name, function, args)
    pipeline = [
        ("1. Ingestion & Sanity-Checks",    run_ingestion,      (input_dir, output_dir)),
        ("2. Authenticity & Anomaly",       run_authenticity,   (input_dir, output_dir)),
        ("3. Metadata Extraction",          run_metadata,       (input_dir, output_dir)),
        ("4. Content Analysis & NLP",       run_content_analysis,(input_dir, output_dir)),
        ("5. Similarity & Correlation",     run_similarity,     (input_dir, output_dir)),
        ("6. ML Fake-vs-Real Classifier",   run_ml_classifier,  (input_dir, output_dir, contamination, random_state)),
        ("7. Generate Charts",              generate_charts,    (input_dir, output_dir)),
        ("8.Reporting & Export",           run_reporting,      (output_dir, output_dir)),
    ]

    timings = {}
    module_summaries = {}

    for name, func, args in pipeline:
        print(f"\n>>> {name}")
        t0 = time.perf_counter()
        try:
            result = func(*args)
            if isinstance(result, dict):
                module_summaries[name] = result
        except Exception as e:
            print(f"[Error] {name} failed: {e}")
            traceback.print_exc()
        elapsed = time.perf_counter() - t0
        timings[name] = elapsed
        print(f"[Timing] {name} completed in {elapsed:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # Write out module timings to JSON (for charting & record)
    timings_path = os.path.join(output_dir, 'module_timings.json')
    try:
        with open(timings_path, 'w', encoding='utf-8') as tf:
            json.dump(timings, tf, indent=2)
        print(f"[Main] Module timings written to {timings_path}")
    except Exception as e:
        print(f"[Main] ERROR writing module_timings.json: {e}")

    total_elapsed = time.perf_counter() - overall_start
    print("\n=== Module Runtimes ===")
    for name, elapsed in timings.items():
        print(f"{name}: {elapsed:.2f}s")
    print(f"Total pipeline time: {total_elapsed:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # File‐type counts
    ingested_json = os.path.join(output_dir, 'ingested_files.json')
    if os.path.exists(ingested_json):
        print("\n=== File Type Counts ===")
        counts = count_file_types(ingested_json)
        # Sort by descending count
        for ext, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{ext}: {cnt}")

    # ──────────────────────────────────────────────────────────────────────────
    # Per‐module summaries
    summary_files = [
        ('ingestion_summary.json',         "Ingestion Summary"),
        ('authenticity_summary.json',      "Authenticity Summary"),
        ('metadata_summary.json',          "Metadata Summary"),
        ('content_analysis_summary.json',  "Content Analysis Summary"),
        ('similarity_summary.json',        "Similarity Summary"),
        ('ml_summary.json',                "ML Classification Summary"),
    ]
    for fname, title in summary_files:
        print_summary(os.path.join(output_dir, fname), title)

    # ──────────────────────────────────────────────────────────────────────────
    # ML Anomaly Rate
    ml_sum_path = os.path.join(output_dir, 'ml_summary.json')
    if os.path.exists(ml_sum_path):
        with open(ml_sum_path, 'r', encoding='utf-8') as f:
            try:
                ml_sum = json.load(f)
                total_files = ml_sum.get('total_files', 0)
                anomalies  = ml_sum.get('anomalies_detected', 0)
                if total_files:
                    pct = anomalies / total_files * 100
                    print(f"\n=== Anomaly Detection ===")
                    print(f"Flagged {anomalies}/{total_files} files ({pct:.2f}%)")
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # Build aggregate_results.json
    agg = {}
    # 1) file_type_counts
    if os.path.exists(ingested_json):
        agg['file_type_counts'] = count_file_types(ingested_json)
    else:
        agg['file_type_counts'] = {}

    # 2) each summary JSON
    for fname, title in summary_files:
        path = os.path.join(output_dir, fname)
        key = title.replace(' ', '_').lower()
        if os.path.exists(path):
            try:
                agg[key] = json.load(open(path, 'r', encoding='utf-8'))
            except Exception:
                agg[key] = None
        else:
            agg[key] = None

    # Write aggregate_results.json
    agg_path = os.path.join(output_dir, 'aggregate_results.json')
    try:
        with open(agg_path, 'w', encoding='utf-8') as f:
            json.dump(agg, f, indent=2)
        print(f"\n✅ Pipeline complete. Aggregate results saved to: {agg_path}")
    except Exception as e:
        print(f"[Main] ERROR writing aggregate_results.json: {e}")

    print(f"🏁 Total runtime: {total_elapsed:.2f}s\n")


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(
        description="Full breach‐analysis pipeline with timing, charts & summaries."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to folder containing breach files (e.g. your “leak” directory)."
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(project_root, 'output'),
        help="Directory where all module outputs (JSON, charts, report) will be written."
    )
    parser.add_argument(
        "-c", "--contamination",
        type=float, default=0.05,
        help="Contamination fraction for IsolationForest (default: 0.05)."
    )
    parser.add_argument(
        "-r", "--random_state",
        type=int, default=42,
        help="Random seed for IsolationForest (default: 42)."
    )
    args = parser.parse_args()

    try:
        main(args.input, args.output, args.contamination, args.random_state)
    except Exception as e:
        print(f"[Fatal] Pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)
