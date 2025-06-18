# modules/reporting.py

import os
import json
from datetime import datetime
from .common_warnings import *    # this sets up filters for PyPDF2
from .pil_config import *        # disables DecompressionBombWarning

def _safe_load_json(path: str):
    """
    Attempt to load a JSON file. If missing or invalid, return None.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def _html_escape(s: str) -> str:
    """
    Simple HTML-escape for text content.
    """
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace("\"", "&quot;")
             .replace("'", "&#39;"))

def _render_list_summary(key: str, val: list, json_path: str) -> str:
    """
    If 'val' is a list, return:
      - If length ≤ 10: comma-separated (escaped) items,
      - Else: "<N items> (see <link to json_path>)"
    """
    length = len(val)
    if length == 0:
        return "<em>0 items</em>"
    if length <= 10:
        # show each item (escaped), separated by commas
        escaped_items = ", ".join(_html_escape(str(item)) for item in val)
        return escaped_items
    else:
        # show "N items" with hyperlink to json_path
        # compute a relative link from the report's folder to the JSON
        filename = os.path.basename(json_path)
        return (f"<em>{length} items</em> "
                f"(<a href=\"{_html_escape(filename)}\" target=\"_blank\">details</a>)")

def _value_to_html(key: str, val, summary_dir: str) -> str:
    """
    Convert a Python value (which may be a dict, list, scalar, etc.)
    into an HTML snippet for a table cell. Lists are treated specially:
      - If 'val' is a list and the summary JSON is in 'summary_dir',
        link to that JSON rather than dump everything.
    """
    # If it's a dict, render recursively as a nested table
    if isinstance(val, dict):
        return _dict_to_html_table(val, summary_dir)

    if isinstance(val, list):
        return _render_list_summary(key, val, os.path.join(summary_dir, f"{key}.json"))

    # Otherwise, just HTML-escape the scalar
    return _html_escape(str(val))

def _dict_to_html_table(d: dict, summary_dir: str) -> str:
    """
    Given a flat dictionary (keys → values), produce an HTML table
    with two columns: “Field” and “Value”. If a value is itself a dict,
    render it recursively as a nested table. If it’s a list, call
    _value_to_html to produce a concise representation.
    """
    rows = []
    for key, val in d.items():
        display_key = _html_escape(str(key))
        display_val = _value_to_html(key, val, summary_dir)
        rows.append(f"""
            <tr>
              <td class="field-cell">{display_key}</td>
              <td class="value-cell">{display_val}</td>
            </tr>
        """)
    table_html = f"""
      <table class="summary-table">
        <thead>
          <tr>
            <th class="field-header">Field</th>
            <th class="value-header">Value</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    """
    return table_html

def _anomalies_to_html_table(anomalies: list) -> str:
    if not anomalies:
        return "<p>No anomalies flagged.</p>"

    limited = anomalies[:10]
    # Collect all possible columns
    all_keys = set()
    for rec in limited:
        all_keys.update(rec.keys())

    # Always show 'relative_path' or 'path' first, then 'anomaly_score'
    preferred_order = []
    if any('path' in r for r in limited):
        preferred_order.append('path')
    elif any('relative_path' in r for r in limited):
        preferred_order.append('relative_path')

    if 'anomaly_score' in all_keys:
        preferred_order.append('anomaly_score')

    # Then any other keys
    for k in sorted(all_keys):
        if k not in preferred_order:
            preferred_order.append(k)

    # Build header row
    header_cells = "".join(f"<th>{_html_escape(str(h))}</th>" for h in preferred_order)

    # Build each row
    row_html_parts = []
    for rec in limited:
        row_cells = []
        for col in preferred_order:
            val = rec.get(col, "")
            row_cells.append(f"<td>{_html_escape(str(val))}</td>")
        row_html_parts.append(f"<tr>{''.join(row_cells)}</tr>")

    table_html = f"""
      <table class="anomaly-table" border="1" cellpadding="4">
        <thead>
          <tr>{header_cells}</tr>
        </thead>
        <tbody>
          {''.join(row_html_parts)}
        </tbody>
      </table>
    """
    return table_html

def run_reporting(input_dir, output_dir, report_filename='report.html'):
    os.makedirs(output_dir, exist_ok=True)

    module_keys = [
        'ingestion',
        'authenticity',
        'metadata',
        'content_analysis',
        'similarity',
        'ml'
    ]
    summaries = {}
    html_sections = []

    # 1) Load each module’s summary JSON
    for key in module_keys:
        summary_path = os.path.join(input_dir, f'{key}_summary.json')
        data = _safe_load_json(summary_path)

        if data is None:
            # If missing, show a warning in the report
            html_sections.append(f"""
              <h2>{key.replace("_", " ").title()} Summary</h2>
              <p class="warning">[Warning] {key}_summary.json not found.</p>
            """)
        else:
            summaries[key] = data
            # Render as HTML table, passing summary_dir so list‐fields can link back
            html_sections.append(f"""
              <h2>{key.replace("_", " ").title()} Summary</h2>
              {_dict_to_html_table(data, input_dir)}
            """)

    # 2) Load top 10 anomalies from ml_records.json
    anomalies = []
    ml_records_path = os.path.join(input_dir, 'ml_records.json')
    raw_ml = _safe_load_json(ml_records_path)
    if raw_ml is not None:
        # We assume each record has “is_anomaly: 1” for anomalies
        flagged = [r for r in raw_ml if r.get('is_anomaly') in (1, True)]
        # Sort by anomaly_score (lowest = most anomalous)
        flagged_sorted = sorted(
            flagged,
            key=lambda x: x.get('anomaly_score', 0)
        )
        anomalies = flagged_sorted[:10]
    summaries['top_anomalies'] = anomalies

    # Render anomalies section
    html_sections.append("<h2>Top 10 Anomalies</h2>")
    if anomalies:
        html_sections.append(_anomalies_to_html_table(anomalies))
    else:
        html_sections.append("<p>No anomalies flagged.</p>")

    charts_dir = os.path.join(output_dir, "charts")

    # 3.1) Module Runtimes Chart
    runtimes_png = os.path.join(charts_dir, "module_runtimes.png")
    if os.path.exists(runtimes_png):
        html_sections.append("""
              <h2>Module Runtimes (seconds)</h2>
              <p>
                <img src="charts/module_runtimes.png"
                     alt="Module Runtimes Chart"
                     style="max-width:100%; height:auto;">
              </p>
            """)
    else:
        html_sections.append("""
              <h2>Module Runtimes (seconds)</h2>
              <p class="warning">Chart not available (module_runtimes.png not found)</p>
            """)

    # 3.2) Entropy Histogram
    entropy_png = os.path.join(charts_dir, "entropy_histogram.png")
    if os.path.exists(entropy_png):
        html_sections.append("""
              <h2>Entropy Distribution (Histogram)</h2>
              <p>
                <img src="charts/entropy_histogram.png"
                     alt="Entropy Histogram"
                     style="max-width:100%; height:auto;">
              </p>
            """)
    else:
        html_sections.append("""
              <h2>Entropy Distribution (Histogram)</h2>
              <p class="warning">Chart not available (entropy_histogram.png not found)</p>
            """)

    # ─────────────────────────────────────────────────────────────────────────

    # 4) Build the final HTML page
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    css = """
      <style>
        body {
          font-family: Arial, sans-serif;
          margin: 20px;
          line-height: 1.4;
        }
        h1 {
          margin-bottom: 0.3em;
        }
        h2 {
          margin-top: 1.5em;
          color: #333333;
        }
        .summary-table, .anomaly-table {
          border-collapse: collapse;
          width: 100%;
          margin-top: 0.5em;
        }
        .summary-table th, .summary-table td,
        .anomaly-table th, .anomaly-table td {
          border: 1px solid #cccccc;
          padding: 8px;
          text-align: left;
        }
        .summary-table tbody tr:nth-child(even),
        .anomaly-table tbody tr:nth-child(even) {
          background-color: #f9f9f9;
        }
        .field-header, .value-header {
          background-color: #f0f0f0;
          font-weight: bold;
        }
        .field-cell {
          width: 30%;
          background-color: #fafafa;
        }
        .value-cell {
          width: 70%;
        }
        .warning {
          color: #990000;
          font-style: italic;
        }
      </style>
    """

    html_parts = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '  <meta charset="utf-8">',
        '  <title>Data Breach Analysis Report</title>',
        css,
        '</head>',
        '<body>',
        f'  <h1>Data Breach Analysis Report</h1>',
        f'  <p><em>Generated on {now}</em></p>'
    ]

    # Inject each module’s HTML section
    html_parts.extend(html_sections)

    html_parts.append('</body></html>')
    report_html = "\n".join(html_parts)

    # 5) Write the HTML report
    report_path = os.path.join(output_dir, report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_html)
    print(f"[Reporting] HTML report written to: {report_path}")

    # 6) Write combined_summary.json (with all summaries + anomalies count)
    combined = summaries.copy()
    combined_path = os.path.join(output_dir, 'combined_summary.json')
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)
    print(f"[Reporting] Combined JSON summary written to: {combined_path}")


if __name__ == '__main__':
    # Auto-detect project root and default output folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_folder = os.path.join(project_root, 'output')

    import argparse
    parser = argparse.ArgumentParser(
        description="Module 7: Reporting & Export"
    )
    parser.add_argument(
        '-i', '--input',
        default=default_folder,
        help=f'Path to folder with *_summary.json and ml_records.json (default: {default_folder})'
    )
    parser.add_argument(
        '-o', '--output',
        default=default_folder,
        help=f'Where to write report.html and combined_summary.json (default: {default_folder})'
    )
    args = parser.parse_args()

    run_reporting(args.input, args.output)
