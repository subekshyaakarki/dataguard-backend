"""
DataGuard — Missing Field Analysis

Goes beyond counting nulls.
Clusters missing values by time period, source, and data provider
to distinguish random noise from systematic pipeline failures.
"""

import pandas as pd
import numpy as np
from typing import Optional


def run_missing_field_check(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    source_col: Optional[str] = None,
    null_threshold: float = 0.05,
) -> dict:
    results = {
        "check": "missing_field_analysis",
        "status": "pass",
        "severity": None,
        "findings": {},
        "business_impact": "",
        "suggested_action": "",
        "cost_of_ignoring": "",
        "fix_code": "",
    }

    total = len(df)
    column_findings = {}
    pipeline_failures = []
    max_null_pct = 0.0

    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = round(null_count / total * 100, 2)

        if null_pct == 0:
            continue

        col_finding = {
            "null_count": int(null_count),
            "null_pct": null_pct,
            "is_systematic": False,
            "systematic_evidence": [],
            "imputation_risk": "low",
        }

        # ── Detect systematic clustering ──────────────────────
        null_mask = df[col].isna()

        # By date
        if date_col and date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df["_dg_period"] = df[date_col].dt.to_period("Q")
                period_null_rates = df.groupby("_dg_period")[col].apply(
                    lambda x: x.isna().mean()
                )
                df.drop(columns=["_dg_period"], inplace=True, errors="ignore")

                # Flag periods with significantly higher null rates
                mean_rate = period_null_rates.mean()
                spiky_periods = period_null_rates[
                    period_null_rates > mean_rate * 3
                ].head(3)

                if len(spiky_periods) > 0 and spiky_periods.max() > 0.3:
                    col_finding["is_systematic"] = True
                    col_finding["systematic_evidence"].append({
                        "type": "date_cluster",
                        "description": f"Null rate spikes in specific time periods: "
                                       f"{', '.join([str(p) for p in spiky_periods.index])}",
                        "max_rate_in_spike": round(float(spiky_periods.max()), 3),
                    })
                    pipeline_failures.append({
                        "column": col,
                        "type": "temporal_cluster",
                        "periods": [str(p) for p in spiky_periods.index],
                    })
            except Exception:
                pass

        # By source
        if source_col and source_col in df.columns:
            source_null_rates = df.groupby(source_col)[col].apply(
                lambda x: x.isna().mean()
            )
            spiky_sources = source_null_rates[source_null_rates > 0.5].head(5)

            if len(spiky_sources) > 0:
                col_finding["is_systematic"] = True
                col_finding["systematic_evidence"].append({
                    "type": "source_cluster",
                    "description": f"High null rates from specific sources: "
                                   f"{', '.join(spiky_sources.index.astype(str).tolist())}",
                    "sources": spiky_sources.index.astype(str).tolist(),
                })
                pipeline_failures.append({
                    "column": col,
                    "type": "source_cluster",
                    "sources": spiky_sources.index.astype(str).tolist(),
                })

        # Imputation risk
        if null_pct > 20:
            col_finding["imputation_risk"] = "high"
        elif null_pct > 10:
            col_finding["imputation_risk"] = "medium"

        if null_pct > max_null_pct:
            max_null_pct = null_pct

        column_findings[col] = col_finding

    # ── Severity ──────────────────────────────────────────────
    high_null_cols = [c for c, v in column_findings.items() if v["null_pct"] > 20]
    systematic_cols = [c for c, v in column_findings.items() if v["is_systematic"]]

    if high_null_cols or len(systematic_cols) >= 2:
        severity = "high"
        status = "fail"
    elif column_findings and max_null_pct > null_threshold * 100:
        severity = "medium"
        status = "warn"
    elif column_findings:
        severity = "low"
        status = "warn"
    else:
        severity = None
        status = "pass"

    results["status"] = status
    results["severity"] = severity
    results["findings"] = {
        "total_records": total,
        "columns_with_nulls": len(column_findings),
        "high_null_columns": high_null_cols,
        "systematic_failure_columns": systematic_cols,
        "pipeline_failures_detected": pipeline_failures,
        "column_details": column_findings,
        "summary": {
            col: {
                "null_pct": v["null_pct"],
                "is_systematic": v["is_systematic"],
                "imputation_risk": v["imputation_risk"],
            }
            for col, v in column_findings.items()
        },
    }

    if status != "pass":
        affected_desc = ", ".join(
            [f"'{c}' ({column_findings[c]['null_pct']}% null)" for c in list(column_findings.keys())[:4]]
        )
        results["business_impact"] = (
            f"Missing values detected in {len(column_findings)} column(s): {affected_desc}. "
            + (
                f"Nulls in {', '.join(systematic_cols)} cluster around specific time periods or sources, "
                f"indicating a pipeline ingestion failure rather than random missing data. "
                if systematic_cols else ""
            )
            + "If these columns are used as training features, your model will silently drop or "
            "mis-impute affected records, introducing systematic bias toward complete records only."
        )
        results["suggested_action"] = (
            (f"For systematic failures ({', '.join(systematic_cols)}): backfill from source system if possible, "
             f"or trace and fix the upstream pipeline failure. "
             if systematic_cols else "")
            + f"For random nulls: impute using mode (categorical) or median (numeric) per relevant segment. "
            + f"For columns with >20% nulls ({', '.join(high_null_cols)}): consider excluding from training features entirely."
            if high_null_cols else ""
        )
        results["cost_of_ignoring"] = (
            "Systematic bias toward records with complete data. Your model underperforms on "
            "the specific user segment or time period affected by the pipeline failure. "
            "This is especially dangerous in regulated industries where demographic completeness "
            "may be correlated with protected characteristics."
        )
        results["fix_code"] = f"""# DataGuard fix — handle missing fields
import pandas as pd
import numpy as np

df = pd.read_csv("your_dataset.csv")

# Columns with missing data: {list(column_findings.keys())[:5]}
print("Null counts:")
print(df[{list(column_findings.keys())[:5]}].isnull().sum())

# Option 1: Backfill systematic failures from source
# (trace pipeline failure in periods: {[pf.get('periods', pf.get('sources', [])) for pf in pipeline_failures[:2]]})

# Option 2: Impute random nulls
for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Option 3: Drop high-null columns from training features
high_null = {high_null_cols}
df_clean = df.drop(columns=[c for c in high_null if c in df.columns], errors='ignore')

df_clean.to_csv("your_dataset_fixed.csv", index=False)
print(f"Saved {{len(df_clean)}} records")
"""

    return results
