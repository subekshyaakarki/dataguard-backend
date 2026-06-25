# DataGuard — Backend

AI data quality checks for ML training pipelines.

## What it does

Runs four automated checks on any CSV dataset before it reaches a training run:

1. **Duplicate detection** — exact and near-duplicate records via hashing and cosine similarity
2. **Label consistency** — finds mislabeled records using TF-IDF + Cleanlab confident learning
3. **Distribution shift** — detects vocabulary drift and statistical drift across time periods
4. **Missing field analysis** — distinguishes random nulls from systematic pipeline failures

Each issue is reported with:
- Business impact (what it will do to your model)
- Suggested action (specific fix, not vague advice)
- Cost of ignoring (what happens if you skip it)
- Fix code (runnable Python you can copy-paste)

## Quick start

```bash
pip install -r requirements.txt
python audit_engine.py        # runs demo audit
```

## Run the API server

```bash
cd api
uvicorn server:app --reload --port 8000
```

Endpoints:
- `POST /audit/upload` — upload CSV and run full audit
- `GET  /audit/{id}` — retrieve full report
- `GET  /audit/{id}/summary` — get findings with business impact
- `GET  /audit/{id}/fix/{check_name}` — get fix code for a specific check

## Run a single audit in Python

```python
import pandas as pd
from audit_engine import run_audit

df = pd.read_csv("your_dataset.csv")
report = run_audit(df, dataset_name="my_dataset")

print(report["health_score"])
# {"score": 61, "rating": "warning", "label": "Needs attention", ...}
```

## Project structure

```
dataguard/
  audit_engine.py          # main orchestrator — runs all checks
  requirements.txt
  checks/
    duplicate_check.py     # exact + near-duplicate detection
    label_check.py         # mislabel detection via cleanlab
    distribution_check.py  # KL divergence drift detection  
    missing_field_check.py # null clustering and pipeline failure detection
  api/
    server.py              # FastAPI REST API
```

## Health score

100 = perfect data. Deductions:
- High severity issue: -20 points
- Medium severity issue: -10 points  
- Low severity issue: -3 points

Score ranges:
- 85–100: Ready to train
- 65–84: Needs attention
- 0–64: Not ready to train

## Privacy

DataGuard processes data in memory only. Raw records are never written to disk
on external servers. A local Python execution option is available — run all checks
in your own environment and only send the summary report.
