# modules/charts.py

import os
import json
import matplotlib.pyplot as plt


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_module_timings(timings_json: str, output_dir: str) -> str:
    """
    Reads module_timings.json (a dict: {module_name: elapsed_seconds,…}),
    makes a horizontal bar chart, and saves it as PNG.

    Returns the path to the PNG file.
    """
    # 1) Load the timings
    with open(timings_json, 'r', encoding='utf-8') as f:
        timings = json.load(f)

    # Sort modules by elapsed time descending (optional)
    sorted_items = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)
    module_names = [nm for nm, _ in sorted_items]
    elapsed_times = [val for _, val in sorted_items]

    # 2) Make a bar chart (ONE FIGURE)
    plt.figure(figsize=(8, 4 + 0.5 * len(module_names)))
    y_positions = range(len(module_names))
    plt.barh(y_positions, elapsed_times)
    plt.yticks(y_positions, module_names)
    plt.xlabel("Elapsed Time (seconds)")
    plt.title("Module Runtimes")
    plt.tight_layout()

    # 3) Save to PNG
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, "module_runtimes.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_entropy_histogram(entropy_json: str, output_dir: str) -> str:
    """
    Reads authenticity_entropy_records.json (a list of {relative_path, entropy}),
    extracts all entropy values, draws a histogram, and saves as PNG.

    Returns the path to the PNG file.
    """
    with open(entropy_json, 'r', encoding='utf-8') as f:
        records = json.load(f)
    entropies = [r.get('entropy', 0.0) for r in records if isinstance(r.get('entropy', None), (int, float))]

    # 2) Plot histogram
    plt.figure(figsize=(6, 4))
    plt.hist(entropies, bins=30, edgecolor='black')
    plt.xlabel("Entropy (bits/byte)")
    plt.ylabel("Number of Files")
    plt.title("Entropy Distribution Across Files")
    plt.tight_layout()

    # 3) Save to PNG
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, "entropy_histogram.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
