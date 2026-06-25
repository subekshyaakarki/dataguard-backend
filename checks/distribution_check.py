"""
DataGuard — Distribution Shift Detection

Compares statistical distributions across time periods or data splits.
Catches vocabulary drift in text data and statistical drift in numeric columns.
Uses KL divergence for distribution comparison.
"""

import pandas as pd
import numpy as np
from typing import Optional
from collections import Counter


def _kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    p = np.array(p, dtype=float) + epsilon
    q = np.array(q, dtype=float) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _vocab_overlap(texts_a: list, texts_b: list) -> dict:
    def tokenize(texts):
        words = set()
        for t in texts:
            words.update(str(t).lower().split())
        return words

    vocab_a = tokenize(texts_a)
    vocab_b = tokenize(texts_b)

    only_in_a = vocab_a - vocab_b
    only_in_b = vocab_b - vocab_a
    shared = vocab_a & vocab_b

    overlap_pct = round(len(shared) / max(len(vocab_a | vocab_b), 1) * 100, 1)

    # Find top new terms in B (production) not in A (training)
    word_freq_b = Counter()
    for t in texts_b:
        word_freq_b.update(str(t).lower().split())

    new_terms = sorted(
        [(w, word_freq_b[w]) for w in only_in_b if word_freq_b[w] > 5],
        key=lambda x: -x[1]
    )[:20]

    return {
        "vocab_size_a": len(vocab_a),
        "vocab_size_b": len(vocab_b),
        "shared_terms": len(shared),
        "only_in_training": len(only_in_a),
        "only_in_production": len(only_in_b),
        "overlap_pct": overlap_pct,
        "new_terms_in_production": [{"term": w, "frequency": f} for w, f in new_terms],
    }


def run_distribution_check(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    text_col: Optional[str] = None,
    split_date: Optional[str] = None,
    kl_threshold: float = 0.3,
) -> dict:
    results = {
        "check": "distribution_shift",
        "status": "pass",
        "severity": None,
        "findings": {},
        "business_impact": "",
        "suggested_action": "",
        "cost_of_ignoring": "",
        "fix_code": "",
    }

    column_results = {}
    max_kl = 0.0
    shift_detected = False

    # ── 1. Split data into two halves ─────────────────────────
    if date_col and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            if split_date:
                cutoff = pd.to_datetime(split_date)
            else:
                cutoff = df[date_col].median()

            df_a = df[df[date_col] < cutoff]
            df_b = df[df[date_col] >= cutoff]
            split_method = f"date column '{date_col}' (cutoff: {cutoff.date()})"
        except Exception:
            mid = len(df) // 2
            df_a, df_b = df.iloc[:mid], df.iloc[mid:]
            split_method = "chronological split (no date column)"
    else:
        mid = len(df) // 2
        df_a, df_b = df.iloc[:mid], df.iloc[mid:]
        split_method = "row-order split (first half vs second half)"

    results["findings"]["split_method"] = split_method
    results["findings"]["split_a_size"] = len(df_a)
    results["findings"]["split_b_size"] = len(df_b)

    if len(df_a) < 10 or len(df_b) < 10:
        results["status"] = "skip"
        results["findings"]["note"] = "Dataset too small to detect distribution shift reliably."
        return results

    # ── 2. Text vocabulary drift ──────────────────────────────
    if text_col and text_col in df.columns:
        texts_a = df_a[text_col].dropna().astype(str).tolist()
        texts_b = df_b[text_col].dropna().astype(str).tolist()

        vocab_stats = _vocab_overlap(texts_a, texts_b)

        # Word frequency KL divergence
        freq_a = Counter()
        freq_b = Counter()
        for t in texts_a:
            freq_a.update(str(t).lower().split())
        for t in texts_b:
            freq_b.update(str(t).lower().split())

        all_words = list(set(freq_a.keys()) | set(freq_b.keys()))
        p = np.array([freq_a.get(w, 0) for w in all_words], dtype=float)
        q = np.array([freq_b.get(w, 0) for w in all_words], dtype=float)
        kl = round(_kl_divergence(p, q), 4)

        vocab_stats["kl_divergence"] = kl
        column_results[text_col] = {
            "type": "text",
            "kl_divergence": kl,
            "drift_detected": kl > kl_threshold,
            **vocab_stats,
        }

        if kl > max_kl:
            max_kl = kl
        if kl > kl_threshold:
            shift_detected = True

    # ── 3. Numeric column drift ───────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if date_col in numeric_cols:
        numeric_cols.remove(date_col)

    for col in numeric_cols[:5]:
        vals_a = df_a[col].dropna().values
        vals_b = df_b[col].dropna().values

        if len(vals_a) < 5 or len(vals_b) < 5:
            continue

        from scipy.stats import ks_2samp
        ks_stat, p_val = ks_2samp(vals_a, vals_b)

        mean_shift = round(float(vals_b.mean() - vals_a.mean()), 4)
        std_shift = round(float(vals_b.std() - vals_a.std()), 4)
        col_drift = p_val < 0.05 and abs(mean_shift) > 0.1 * abs(vals_a.mean() + 1e-10)

        column_results[col] = {
            "type": "numeric",
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 4),
            "mean_shift": mean_shift,
            "std_shift": std_shift,
            "drift_detected": col_drift,
        }
        if col_drift:
            shift_detected = True

    # ── 4. Categorical column drift ───────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    skip_cols = [text_col, date_col] if text_col and date_col else [text_col or date_col]
    cat_cols = [c for c in cat_cols if c not in skip_cols][:3]

    for col in cat_cols:
        cats_a = df_a[col].value_counts(normalize=True)
        cats_b = df_b[col].value_counts(normalize=True)
        all_cats = list(set(cats_a.index) | set(cats_b.index))

        p = np.array([cats_a.get(c, 0) for c in all_cats])
        q = np.array([cats_b.get(c, 0) for c in all_cats])
        kl = round(_kl_divergence(p, q), 4)

        new_cats = [c for c in cats_b.index if c not in cats_a.index]
        gone_cats = [c for c in cats_a.index if c not in cats_b.index]

        column_results[col] = {
            "type": "categorical",
            "kl_divergence": kl,
            "drift_detected": kl > kl_threshold,
            "new_categories": new_cats[:10],
            "disappeared_categories": gone_cats[:10],
        }
        if kl > max_kl:
            max_kl = kl
        if kl > kl_threshold:
            shift_detected = True

    # ── 5. Severity ───────────────────────────────────────────
    if max_kl > 0.8 or (shift_detected and len([c for c in column_results.values() if c.get("drift_detected")]) >= 3):
        severity = "high"
        status = "fail"
    elif shift_detected:
        severity = "medium"
        status = "warn"
    else:
        severity = None
        status = "pass"

    drifted_cols = [k for k, v in column_results.items() if v.get("drift_detected")]

    results["status"] = status
    results["severity"] = severity
    results["findings"] = {
        "max_kl_divergence": round(max_kl, 4),
        "kl_threshold": kl_threshold,
        "shift_detected": shift_detected,
        "columns_with_drift": drifted_cols,
        "column_details": column_results,
    }

    if status != "pass":
        results["business_impact"] = (
            f"Significant distribution shift detected in {len(drifted_cols)} column(s): {', '.join(drifted_cols)}. "
            f"Your model will be trained on patterns that no longer reflect current data. "
            f"It will perform well on historical test sets but underperform on real-world inputs "
            f"that reflect the newer distribution."
        )
        results["suggested_action"] = (
            "Option 1: Retrain on only recent data that matches current distribution. "
            "Option 2: Run a normalization pass to map old terminology/values to current schema. "
            "Option 3: Weight recent data more heavily during training. "
            f"Start with the column showing most drift: '{drifted_cols[0] if drifted_cols else 'text column'}'."
        )
        results["cost_of_ignoring"] = (
            "Strong benchmark numbers will mask weak production performance. "
            "The gap between test accuracy and real-world accuracy widens as your "
            "product evolves and training data ages. Difficult to diagnose without "
            "explicit drift analysis."
        )
        results["fix_code"] = f"""# DataGuard fix — address distribution shift
import pandas as pd

df = pd.read_csv("your_dataset.csv")

# Option 1: Use only recent data
# Columns with detected drift: {drifted_cols}
df['{date_col or "created_at"}'] = pd.to_datetime(df['{date_col or "created_at"}'])
cutoff = pd.Timestamp('2024-01-01')
df_recent = df[df['{date_col or "created_at"}'] >= cutoff]
print(f"Records after date filter: {{len(df_recent)}}")

# Option 2: Check distribution of drifted columns
for col in {drifted_cols}:
    print(f"\\n{{col}} value distribution:")
    print(df_recent[col].value_counts().head(10))

df_recent.to_csv("your_dataset_recent.csv", index=False)
"""

    return results
