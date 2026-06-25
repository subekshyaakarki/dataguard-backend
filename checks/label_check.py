"""
DataGuard — Label Consistency Check

Uses TF-IDF + Logistic Regression with cleanlab's confident learning
to find records where a cross-validated model disagrees with the
assigned label at high confidence.

Falls back to TF-IDF when sentence-transformers are unavailable.
Human always makes the final call — DataGuard never auto-relabels.
"""

import pandas as pd
import numpy as np
from typing import Optional


def run_label_check(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    confidence_threshold: float = 0.75,
) -> dict:
    results = {
        "check": "label_consistency",
        "status": "pass",
        "severity": None,
        "findings": {},
        "business_impact": "",
        "suggested_action": "",
        "cost_of_ignoring": "",
        "fix_code": "",
    }

    if text_col not in df.columns or label_col not in df.columns:
        results["status"] = "error"
        results["findings"]["error"] = f"Columns '{text_col}' or '{label_col}' not found."
        return results

    total = len(df)
    labels = df[label_col].astype(str)
    unique_labels = labels.unique().tolist()

    if len(unique_labels) < 2:
        results["status"] = "skip"
        results["findings"]["note"] = "Need at least 2 unique labels to run consistency check."
        return results

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import cross_val_predict
        from cleanlab.filter import find_label_issues

        sample_df = df[[text_col, label_col]].dropna().copy()
        if len(sample_df) > 5000:
            sample_df = sample_df.sample(5000, random_state=42)

        texts = sample_df[text_col].astype(str).tolist()
        raw_labels = sample_df[label_col].astype(str).tolist()

        le = LabelEncoder()
        encoded_labels = le.fit_transform(raw_labels)

        # TF-IDF embeddings — no network calls needed
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
        X = vectorizer.fit_transform(texts)

        clf = LogisticRegression(max_iter=500, random_state=42, C=1.0)
        n_splits = min(5, len(set(encoded_labels)))
        pred_probs = cross_val_predict(
            clf, X, encoded_labels,
            cv=n_splits, method="predict_proba"
        )

        issue_indices = find_label_issues(
            labels=encoded_labels,
            pred_probs=pred_probs,
            return_indices_ranked_by="self_confidence",
        )

        flagged = []
        for idx in issue_indices:
            original_idx = int(sample_df.index[idx])
            actual_label = raw_labels[idx]
            pred_label_idx = int(np.argmax(pred_probs[idx]))
            pred_label = le.inverse_transform([pred_label_idx])[0]
            confidence = round(float(pred_probs[idx][pred_label_idx]), 3)

            if confidence >= confidence_threshold:
                flagged.append({
                    "record_index": original_idx,
                    "current_label": actual_label,
                    "suggested_label": pred_label,
                    "model_confidence": confidence,
                    "text_preview": str(texts[idx])[:150],
                })

        flagged_count = len(flagged)
        flagged_pct = round(flagged_count / total * 100, 1)

        class_breakdown = {}
        for item in flagged:
            lbl = item["current_label"]
            class_breakdown[lbl] = class_breakdown.get(lbl, 0) + 1

        if flagged_pct >= 5:
            severity = "high"; status = "fail"
        elif flagged_pct >= 2:
            severity = "medium"; status = "warn"
        elif flagged_pct > 0:
            severity = "low"; status = "warn"
        else:
            severity = None; status = "pass"

        results.update({
            "status": status,
            "severity": severity,
            "findings": {
                "total_records": total,
                "flagged_count": flagged_count,
                "flagged_pct": flagged_pct,
                "class_breakdown": class_breakdown,
                "top_flagged_examples": flagged[:10],
                "unique_labels": unique_labels,
                "confidence_threshold": confidence_threshold,
                "method": "TF-IDF + Logistic Regression + Cleanlab confident learning",
            }
        })

        if status != "pass":
            results["business_impact"] = (
                f"{flagged_count:,} records ({flagged_pct}% of dataset) may have incorrect labels. "
                f"Your model is learning from examples where the assigned label contradicts the "
                f"patterns in the content. This corrupts the decision boundaries your model will "
                f"apply in production."
            )
            results["suggested_action"] = (
                f"Review the top {min(200, flagged_count)} flagged records. "
                f"Re-label correct examples, exclude genuinely ambiguous ones, and retrain. "
                f"Classes most affected: "
                f"{', '.join([f'{k} ({v} records)' for k, v in sorted(class_breakdown.items(), key=lambda x: -x[1])[:3]])}"
            )
            results["cost_of_ignoring"] = (
                "Model learns incorrect decision logic. Hard to diagnose post-deployment "
                "because surface metrics appear normal until it fails on specific subclasses."
            )
            results["fix_code"] = f"""# DataGuard fix — review and re-label flagged records
import pandas as pd

df = pd.read_csv("your_dataset.csv")

flagged_indices = {[f['record_index'] for f in flagged[:20]]}
flagged_df = df.loc[flagged_indices].copy()
flagged_df['dataguard_suggested_label'] = {[f['suggested_label'] for f in flagged[:20]]}
flagged_df['dataguard_confidence'] = {[f['model_confidence'] for f in flagged[:20]]}

flagged_df.to_csv("flagged_for_review.csv", index=True)
print(f"Exported {{len(flagged_df)}} records for human review")
"""

    except Exception as e:
        results["status"] = "error"
        results["findings"]["error"] = str(e)

    return results
