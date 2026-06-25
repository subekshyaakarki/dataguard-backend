"""
DataGuard — Main Audit Engine

Orchestrates all four checks, computes the health score,
and produces a structured report with business impact framing.
"""

import pandas as pd
import numpy as np
import json
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from checks.duplicate_check import run_duplicate_check
from checks.label_check import run_label_check
from checks.distribution_check import run_distribution_check
from checks.missing_field_check import run_missing_field_check


def compute_health_score(check_results: list) -> dict:
    """
    Score: 100 = perfect. Each high issue -20, medium -10, low -3.
    """
    score = 100
    deductions = []

    severity_map = {"high": 20, "medium": 10, "low": 3}

    for check in check_results:
        sev = check.get("severity")
        if sev in severity_map:
            deduction = severity_map[sev]
            score -= deduction
            deductions.append({
                "check": check["check"],
                "severity": sev,
                "deduction": deduction,
            })

    score = max(0, score)

    if score >= 85:
        rating = "good"
        label = "Ready to train"
    elif score >= 65:
        rating = "warning"
        label = "Needs attention"
    else:
        rating = "critical"
        label = "Not ready to train"

    return {
        "score": score,
        "rating": rating,
        "label": label,
        "deductions": deductions,
    }


def detect_columns(df: pd.DataFrame) -> dict:
    """Auto-detect likely text, label, date, and id columns."""
    cols = {}

    # Text column — longest avg string length
    str_cols = df.select_dtypes(include=["object"]).columns
    if len(str_cols) > 0:
        avg_lens = {c: df[c].dropna().astype(str).str.len().mean() for c in str_cols}
        cols["text_col"] = max(avg_lens, key=avg_lens.get)

    # Label column — low cardinality string column
    low_card = [
        c for c in str_cols
        if c != cols.get("text_col")
        and df[c].nunique() <= 20
        and df[c].nunique() >= 2
    ]
    if low_card:
        cols["label_col"] = low_card[0]

    # Date column
    for col in df.columns:
        if any(kw in col.lower() for kw in ["date", "time", "created", "updated", "at"]):
            try:
                pd.to_datetime(df[col].dropna().head(10))
                cols["date_col"] = col
                break
            except Exception:
                pass

    # ID column
    for col in df.columns:
        if any(kw in col.lower() for kw in ["id", "_id", "uuid", "key"]):
            cols["id_col"] = col
            break

    return cols


def run_audit(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    date_col: Optional[str] = None,
    id_col: Optional[str] = None,
    source_col: Optional[str] = None,
    run_parallel: bool = True,
) -> dict:
    start_time = time.time()

    # Auto-detect columns if not provided
    detected = detect_columns(df)
    text_col = text_col or detected.get("text_col")
    label_col = label_col or detected.get("label_col")
    date_col = date_col or detected.get("date_col")
    id_col = id_col or detected.get("id_col")

    print(f"\nDataGuard audit starting: {dataset_name}")
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Auto-detected — text: {text_col}, label: {label_col}, date: {date_col}")
    print()

    check_fns = {
        "duplicate_detection": lambda: run_duplicate_check(
            df.copy(), text_col=text_col, id_col=id_col
        ),
        "label_consistency": lambda: run_label_check(
            df.copy(), text_col=text_col, label_col=label_col
        ) if text_col and label_col else {"check": "label_consistency", "status": "skip", "findings": {"note": "No text+label columns found"}},
        "distribution_shift": lambda: run_distribution_check(
            df.copy(), date_col=date_col, text_col=text_col
        ),
        "missing_field_analysis": lambda: run_missing_field_check(
            df.copy(), date_col=date_col, source_col=source_col
        ),
    }

    check_results = {}

    if run_parallel:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fn): name for name, fn in check_fns.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    check_results[name] = result
                    status_icon = {"pass": "✓", "warn": "⚠", "fail": "✗", "skip": "—", "error": "!"}.get(result.get("status"), "?")
                    print(f"  [{status_icon}] {name}: {result.get('status', 'unknown')}")
                except Exception as e:
                    check_results[name] = {"check": name, "status": "error", "findings": {"error": str(e)}}
                    print(f"  [!] {name}: error — {e}")
    else:
        for name, fn in check_fns.items():
            print(f"  Running {name}...")
            try:
                result = fn()
                check_results[name] = result
                status_icon = {"pass": "✓", "warn": "⚠", "fail": "✗", "skip": "—"}.get(result.get("status"), "?")
                print(f"  [{status_icon}] {name}: {result.get('status')}")
            except Exception as e:
                check_results[name] = {"check": name, "status": "error", "findings": {"error": str(e)}}

    checks_list = list(check_results.values())
    health = compute_health_score(checks_list)
    elapsed = round(time.time() - start_time, 1)

    issues_count = {
        "high": sum(1 for c in checks_list if c.get("severity") == "high"),
        "medium": sum(1 for c in checks_list if c.get("severity") == "medium"),
        "low": sum(1 for c in checks_list if c.get("severity") == "low"),
    }

    report = {
        "dataset_name": dataset_name,
        "records_audited": len(df),
        "columns_audited": list(df.columns),
        "audit_duration_seconds": elapsed,
        "health_score": health,
        "issues_summary": issues_count,
        "detected_columns": {
            "text_col": text_col,
            "label_col": label_col,
            "date_col": date_col,
            "id_col": id_col,
        },
        "checks": check_results,
    }

    print(f"\n{'─'*50}")
    print(f"DataGuard audit complete in {elapsed}s")
    print(f"Health score: {health['score']}/100 — {health['label']}")
    print(f"Issues: {issues_count['high']} high · {issues_count['medium']} medium · {issues_count['low']} low")

    return report


def print_report(report: dict):
    """Pretty-print the audit report to terminal."""
    h = report["health_score"]
    print(f"\n{'═'*60}")
    print(f"  DataGuard Audit Report — {report['dataset_name']}")
    print(f"{'═'*60}")
    print(f"  Records audited: {report['records_audited']:,}")
    print(f"  Duration: {report['audit_duration_seconds']}s")
    print(f"  Health score: {h['score']}/100 — {h['label']}")
    print()

    for check_name, check in report["checks"].items():
        status = check.get("status", "unknown")
        sev = check.get("severity", "")
        icon = {"pass": "✓", "warn": "⚠", "fail": "✗", "skip": "—", "error": "!"}.get(status, "?")
        sev_str = f" [{sev.upper()}]" if sev else ""
        print(f"  {icon} {check_name.replace('_', ' ').title()}{sev_str}")

        if check.get("business_impact"):
            print(f"    Impact:  {check['business_impact'][:120]}...")
        if check.get("suggested_action"):
            print(f"    Action:  {check['suggested_action'][:120]}...")
        print()


if __name__ == "__main__":
    # ── Demo: generate synthetic dataset and run full audit ───
    import random

    random.seed(42)
    np.random.seed(42)

    n = 500
    labels = ["high_risk", "low_risk", "medium_risk"]

    texts = [
        f"Patient presented with {'chest pain' if i % 3 == 0 else 'routine checkup' if i % 3 == 1 else 'shortness of breath'}, "
        f"age {random.randint(25, 85)}, {'referred to cardiology' if i % 4 == 0 else 'no concerns noted'}."
        for i in range(n)
    ]

    # Introduce duplicates (~20%)
    for i in range(int(n * 0.2)):
        idx = random.randint(0, n - 1)
        texts.append(texts[idx])
        labels_ext = labels[:]

    label_list = [random.choice(labels) for _ in range(n)]

    # Introduce some label noise
    for i in range(int(n * 0.08)):
        idx = random.randint(0, n - 1)
        label_list[idx] = random.choice([l for l in labels if l != label_list[idx]])

    import pandas as pd
    from datetime import datetime, timedelta

    dates = [datetime(2022, 1, 1) + timedelta(days=random.randint(0, 900)) for _ in range(n)]

    df_demo = pd.DataFrame({
        "patient_id": [f"PT-{1000+i}" for i in range(n)],
        "note_text": texts[:n],
        "label": label_list,
        "created_at": dates,
        "department": [random.choice(["cardiology", "general", "emergency", None]) for _ in range(n)],
    })

    # Add some nulls to department
    null_mask = (pd.Series(dates) > datetime(2023, 1, 1)) & (pd.Series(dates) < datetime(2023, 6, 1))
    df_demo.loc[null_mask.values, "department"] = None

    print(f"Demo dataset: {len(df_demo)} records")
    print(df_demo.head(3).to_string())

    report = run_audit(
        df_demo,
        dataset_name="patient_support_demo",
        run_parallel=False,
    )

    print_report(report)

    with open("/mnt/user-data/outputs/dataguard_audit_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nFull report saved to dataguard_audit_report.json")
