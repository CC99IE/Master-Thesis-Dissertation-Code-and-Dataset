# modules/ml_classifier.py

import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from PIL import Image, ExifTags
Image.MAX_IMAGE_PIXELS = None
def load_json(path: str) -> pd.DataFrame:
    """
    Load a JSON list of records from `path` into a pandas DataFrame.
    If the file does not exist or is empty, returns an empty DataFrame.
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # If data is a list of dicts, this builds a DataFrame; otherwise, try to coerce
    try:
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

def run_ml_classifier(input_dir: str,
                      output_dir: str,
                      contamination: float = 0.05,
                      random_state: int = 42) -> dict:
    """
    Fake‐vs‐Real Classifier (unsupervised).

    This function:
      1. Reads previous‐module JSON outputs from `input_dir`:
         - ingested_files.json
         - authenticity_entropy_records.json
         - metadata_records.json
         - content_analysis_records.json
         - text_similarity.json
         - image_similarity.json
      2. Builds a per‐file feature matrix that may include:
         - size_bytes             (from ingestion)
         - entropy                (from authenticity checks)
         - metadata_fields_count  (number of metadata fields extracted)
         - text_length, n_entities (from content analysis)
         - duplicate_count        (text‐similarity count)
         - image_duplicate_count  (image‐similarity count)
      3. Scales these features with StandardScaler.
      4. Trains an IsolationForest (with the given contamination & random_state).
      5. Flags each file as anomaly or not, and writes two outputs:
         - ml_records.json  (one record per file, with features + anomaly_score + is_anomaly)
         - ml_summary.json  (aggregate stats: counts, means, stds, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ————— 1) Load ingestion results (required) —————
    ingest_path = os.path.join(output_dir, 'ingested_files.json')
    if not os.path.exists(ingest_path):
        raise FileNotFoundError(f"Cannot find ingestion results at: {ingest_path}")
    df_ing = load_json(ingest_path)
    # Expect at least columns: 'relative_path' and 'size_bytes'
    if 'relative_path' not in df_ing.columns:
        # Maybe older version used 'path'? Rename if present.
        if 'path' in df_ing.columns:
            df_ing = df_ing.rename(columns={'path': 'relative_path'})
        else:
            raise KeyError("ingested_files.json must contain a 'relative_path' field.")
    if 'size_bytes' not in df_ing.columns:
        # If ingestion didn’t record size_bytes, create a zero‐filled column
        df_ing['size_bytes'] = 0

    # ————— 2) Load authenticity (entropy) results  —————
    auth_path = os.path.join(output_dir, 'authenticity_entropy_records.json')
    df_auth = load_json(auth_path)
    # Expect columns: 'relative_path' and 'entropy'
    if 'relative_path' in df_auth.columns:
        # Rename if needed
        if 'path' in df_auth.columns and 'relative_path' not in df_auth.columns:
            df_auth = df_auth.rename(columns={'path': 'relative_path'})
    else:
        # No authenticity file or missing columns → empty DataFrame with proper cols
        df_auth = pd.DataFrame(columns=['relative_path', 'entropy'])

    # ————— 3) Load metadata —————
    meta_path = os.path.join(output_dir, 'metadata_records.json')
    df_meta_raw = load_json(meta_path)
    # Expect records like { 'relative_path': ..., 'metadata': {...}, [ 'error': ... ] }
    if 'relative_path' in df_meta_raw.columns and 'metadata' in df_meta_raw.columns:
        # Compute how many metadata fields each file had
        def _count_fields(m):
            if isinstance(m, dict):
                return len(m)
            return 0
        df_meta_raw['metadata_fields_count'] = df_meta_raw['metadata'].apply(_count_fields)
        df_meta = df_meta_raw[['relative_path', 'metadata_fields_count']]
    else:
        df_meta = pd.DataFrame(columns=['relative_path', 'metadata_fields_count'])

    # ————— 4) Load content analysis —————
    content_path = os.path.join(output_dir, 'content_analysis_records.json')
    df_content = load_json(content_path)
    # Expect at least 'relative_path', 'text_length', 'n_entities'
    if 'relative_path' in df_content.columns:
        # Ensure those columns exist (else create zero‐filled)
        for col in ('text_length', 'n_entities'):
            if col not in df_content.columns:
                df_content[col] = 0
        df_content = df_content[['relative_path', 'text_length', 'n_entities']]
    else:
        df_content = pd.DataFrame(columns=['relative_path', 'text_length', 'n_entities'])

    # ————— 5) Load text similarity —————
    # The file text_similarity.json is assumed to be a list of records
    #   [ { 'file1': 'a.txt', 'file2': 'b.txt', 'cosine_similarity': 0.87 }, ... ]
    # We count, for each file, how many times it appears in either file1 or file2.
    sim_text_path = os.path.join(output_dir, 'text_similarity.json')
    if os.path.exists(sim_text_path):
        df_sim_text_raw = load_json(sim_text_path)
        counts_text = {}
        for _, row in df_sim_text_raw.iterrows():
            f1 = row.get('file1') or row.get('relative_path')  # fallback if named differently
            f2 = row.get('file2')
            if pd.notna(f1):
                counts_text[f1] = counts_text.get(f1, 0) + 1
            if pd.notna(f2):
                counts_text[f2] = counts_text.get(f2, 0) + 1
        df_sim_text = pd.DataFrame(
            list(counts_text.items()),
            columns=['relative_path', 'duplicate_count']
        )
    else:
        df_sim_text = pd.DataFrame(columns=['relative_path', 'duplicate_count'])

    # ————— 6) Load image similarity —————
    # The file image_similarity.json is assumed to be a list of records
    #   [ { 'file1': 'x.jpg', 'file2': 'y.jpg', 'hamming_distance': 4 }, ... ]
    counts_img = {}
    sim_img_path = os.path.join(output_dir, 'image_similarity.json')
    if os.path.exists(sim_img_path):
        df_sim_img_raw = load_json(sim_img_path)
        for _, row in df_sim_img_raw.iterrows():
            f1 = row.get('file1')
            f2 = row.get('file2')
            if pd.notna(f1):
                counts_img[f1] = counts_img.get(f1, 0) + 1
            if pd.notna(f2):
                counts_img[f2] = counts_img.get(f2, 0) + 1
        df_sim_img = pd.DataFrame(
            list(counts_img.items()),
            columns=['relative_path', 'image_duplicate_count']
        )
    else:
        df_sim_img = pd.DataFrame(columns=['relative_path', 'image_duplicate_count'])

    # ————— 7) Merge everything into a single DataFrame —————
    df = df_ing.copy()

    # Helper: if any other DF has a "path" instead of "relative_path", rename it
    for other_df in (df_auth, df_meta, df_content, df_sim_text, df_sim_img):
        if 'path' in other_df.columns and 'relative_path' not in other_df.columns:
            other_df.rename(columns={'path': 'relative_path'}, inplace=True)

    # Merge step by step, left‐joining on 'relative_path'
    def _merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        return pd.merge(
            left, right,
            on='relative_path',
            how='left'
        )

    df = _merge(df, df_auth[['relative_path', 'entropy']])
    df = _merge(df, df_meta[['relative_path', 'metadata_fields_count']])
    df = _merge(df, df_content[['relative_path', 'text_length', 'n_entities']])
    df = _merge(df, df_sim_text[['relative_path', 'duplicate_count']])
    df = _merge(df, df_sim_img[['relative_path', 'image_duplicate_count']])

    # ————— 8) Ensure all numeric feature columns exist, filling NaN with 0 —————
    numeric_cols = [
        'size_bytes',               # from ingestion
        'entropy',                  # from authenticity
        'metadata_fields_count',    # from metadata
        'text_length',              # from content analysis
        'n_entities',               # from content analysis
        'duplicate_count',          # from text similarity
        'image_duplicate_count'     # from image similarity
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        # Coerce to float and fill NaN
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    # ————— 9) Build feature matrix & scale —————
    feature_cols = numeric_cols.copy()
    X = df[feature_cols].values  # shape (N, 7)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ————— 10) Train IsolationForest —————
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )
    preds = iso.fit_predict(X_scaled)                 # -1 says “anomaly”, +1 says “normal”
    scores = iso.decision_function(X_scaled)           # higher = more “normal”

    df['anomaly_score'] = scores
    df['is_anomaly'] = (preds == -1).astype(int)       # 1 = anomaly, 0 = normal

    # ————— 11) Write per‐file records to ml_records.json —————
    ml_records = []
    for _, row in df.iterrows():
        rec = {'relative_path': row['relative_path']}
        for col in feature_cols:
            rec[col] = float(row[col])
        rec['anomaly_score'] = float(row['anomaly_score'])
        rec['is_anomaly'] = int(row['is_anomaly'])
        ml_records.append(rec)

    ml_records_path = os.path.join(output_dir, 'ml_records.json')
    with open(ml_records_path, 'w', encoding='utf-8') as f:
        json.dump(ml_records, f, indent=2)

    # ————— 12) Write summary to ml_summary.json —————
    summary = {
        'total_files': int(df.shape[0]),
        'anomalies_detected': int(df['is_anomaly'].sum()),
        'contamination': contamination,
        'feature_means': {col: float(df[col].mean()) for col in feature_cols},
        'feature_std': {col: float(df[col].std()) for col in feature_cols},
        'ml_records_path': ml_records_path
    }
    ml_summary_path = os.path.join(output_dir, 'ml_summary.json')
    with open(ml_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary


# Optional: allow this module to be run standalone from the command line
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Module 6: Fake-vs-Real classifier (unsupervised)"
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='Path to the folder containing previous modules’ JSON outputs'
    )
    parser.add_argument(
        '-o', '--output', required=True,
        help='Path to write ml_records.json and ml_summary.json'
    )
    parser.add_argument(
        '-c', '--contamination', type=float, default=0.05,
        help='Expected proportion of anomalies (float in 0–0.5).'
    )
    parser.add_argument(
        '-r', '--random_state', type=int, default=42,
        help='Random seed for IsolationForest.'
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    result = run_ml_classifier(
        input_dir=args.input,
        output_dir=args.output,
        contamination=args.contamination,
        random_state=args.random_state
    )
    print(f"[ML] Done. Found {result['anomalies_detected']} anomalies "
          f"out of {result['total_files']} files.")
