"""
DataGuard — Duplicate Detection Check

Catches two types of duplicates:
1. Exact duplicates — identical rows via hash
2. Near-duplicates — semantically similar text via cosine similarity
"""

import pandas as pd
import numpy as np
import hashlib
from typing import Optional


def run_duplicate_check(
    df: pd.DataFrame,
    text_col: Optional[str] = None,
    id_col: Optional[str] = None,
    similarity_threshold: float = 0.92,
) -> dict:
    results = {
        "check": "duplicate_detection",
        "status": "pass",
        "severity": None,
        "findings": {},
        "business_impact": "",
        "suggested_action": "",
        "cost_of_ignoring": "",
        "fix_code": "",
    }

    total = len(df)

    # ── 1. Exact duplicates ───────────────────────────────────
    exact_dupes = df[df.duplicated(keep=False)]
    exact_count = len(exact_dupes)
    exact_pct = round(exact_count / total * 100, 1)

    # ── 2. Near-duplicate detection on text column ────────────
    near_dupe_pairs = []
    near_dupe_indices = set()

    if text_col and text_col in df.columns:
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity

            sample = df[text_col].fillna("").astype(str)
            # Cap at 2000 rows for speed — sample if larger
            if len(sample) > 2000:
                sample = sample.sample(2000, random_state=42)

            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(sample.tolist(), show_progress_bar=False)
            sim_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(sim_matrix, 0)

            pairs = np.argwhere(sim_matrix > similarity_threshold)
            # Deduplicate pairs (i,j) and (j,i)
            seen = set()
            for i, j in pairs:
                key = tuple(sorted([int(i), int(j)]))
                if key not in seen:
                    seen.add(key)
                    near_dupe_pairs.append({
                        "index_a": int(sample.index[i]),
                        "index_b": int(sample.index[j]),
                        "similarity": round(float(sim_matrix[i][j]), 3),
                        "text_a": str(sample.iloc[i])[:120],
                        "text_b": str(sample.iloc[j])[:120],
                    })
                    near_dupe_indices.update([int(sample.index[i]), int(sample.index[j])])

        except Exception as e:
            results["findings"]["near_duplicate_error"] = str(e)

    near_dupe_count = len(near_dupe_indices)
    near_dupe_pct = round(near_dupe_count / total * 100, 1)

    total_affected = len(set(exact_dupes.index) | near_dupe_indices)
    total_pct = round(total_affected / total * 100, 1)

    # ── 3. Severity scoring ───────────────────────────────────
    if total_pct >= 20:
        severity = "high"
        status = "fail"
    elif total_pct >= 5:
        severity = "medium"
        status = "warn"
    elif total_pct > 0:
        severity = "low"
        status = "warn"
    else:
        severity = None
        status = "pass"

    results["status"] = status
    results["severity"] = severity
    results["findings"] = {
        "total_records": total,
        "exact_duplicates": exact_count,
        "exact_duplicate_pct": exact_pct,
        "near_duplicate_records": near_dupe_count,
        "near_duplicate_pct": near_dupe_pct,
        "total_affected": total_affected,
        "total_affected_pct": total_pct,
        "near_duplicate_examples": near_dupe_pairs[:5],
    }

    if status != "pass":
        results["business_impact"] = (
            f"{total_pct}% of your dataset contains duplicate or near-duplicate records. "
            f"Your model will see these examples disproportionately during training, "
            f"causing it to overweight those patterns and become overconfident on cases "
            f"that may not reflect real-world distribution."
        )
        results["suggested_action"] = (
            f"Deduplicate your dataset. "
            + (f"For exact duplicates: drop rows with identical content. " if exact_count > 0 else "")
            + (f"For near-duplicates: review the {len(near_dupe_pairs)} flagged pairs and remove the lower-quality copy. " if near_dupe_pairs else "")
            + f"Estimated clean records after fix: ~{total - total_affected:,}."
        )
        results["cost_of_ignoring"] = (
            "Model confidence scores will be systematically inflated on duplicated patterns. "
            "Eval metrics will look better than production performance because test overlap "
            "with duplicated training examples inflates accuracy."
        )

        dedup_cols = [id_col, text_col] if id_col and text_col else (
            [text_col] if text_col else list(df.columns)
        )
        col_str = str(dedup_cols)
        results["fix_code"] = f"""# DataGuard fix — remove duplicate records
import pandas as pd

df = pd.read_csv("your_dataset.csv")

# Step 1: Remove exact duplicates
before = len(df)
df_clean = df.drop_duplicates(subset={col_str}, keep='first')
print(f"Exact duplicates removed: {{before - len(df_clean)}}")

# Step 2: For near-duplicates, review flagged pairs first
# Near-duplicate pairs flagged by DataGuard:
flagged_pairs = {near_dupe_pairs[:3]}

# Save clean dataset
df_clean.to_csv("your_dataset_clean.csv", index=False)
print(f"Clean records: {{len(df_clean)}}")
"""

    return results
