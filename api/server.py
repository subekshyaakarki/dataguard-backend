"""
DataGuard — FastAPI Backend

Endpoints:
  POST /audit/upload   — upload CSV and run full audit
  POST /audit/run      — run audit on JSON data
  GET  /audit/{id}     — retrieve audit report
  GET  /health         — service health check
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import uuid
import time
from io import StringIO
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from audit_engine import run_audit

app = FastAPI(
    title="DataGuard API",
    description="AI data quality checks for ML training pipelines",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

audit_store = {}


class AuditConfig(BaseModel):
    dataset_name: str = "dataset"
    text_col: Optional[str] = None
    label_col: Optional[str] = None
    date_col: Optional[str] = None
    id_col: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "DataGuard API", "version": "1.0.0"}


@app.post("/audit/upload")
async def upload_and_audit(
    file: UploadFile = File(...),
    dataset_name: str = "dataset",
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    date_col: Optional[str] = None,
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files supported. JSON coming soon.")

    content = await file.read()
    try:
        df = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if len(df) == 0:
        raise HTTPException(400, "Dataset is empty.")

    if len(df) > 100_000:
        raise HTTPException(400, "Dataset too large for free tier. Max 100,000 rows.")

    audit_id = str(uuid.uuid4())[:8]

    try:
        report = run_audit(
            df,
            dataset_name=dataset_name or file.filename,
            text_col=text_col,
            label_col=label_col,
            date_col=date_col,
            run_parallel=False,
        )
        report["audit_id"] = audit_id
        audit_store[audit_id] = report
    except Exception as e:
        raise HTTPException(500, f"Audit failed: {e}")

    return {
        "audit_id": audit_id,
        "status": "complete",
        "health_score": report["health_score"],
        "issues_summary": report["issues_summary"],
        "records_audited": report["records_audited"],
        "message": f"Audit complete. Health score: {report['health_score']['score']}/100 — {report['health_score']['label']}",
    }


@app.get("/audit/{audit_id}")
def get_audit(audit_id: str):
    if audit_id not in audit_store:
        raise HTTPException(404, f"Audit '{audit_id}' not found.")
    return audit_store[audit_id]


@app.get("/audit/{audit_id}/summary")
def get_audit_summary(audit_id: str):
    if audit_id not in audit_store:
        raise HTTPException(404, "Audit not found.")

    report = audit_store[audit_id]
    checks = report["checks"]

    findings = []
    for check_name, check in checks.items():
        if check.get("severity"):
            findings.append({
                "check": check_name.replace("_", " ").title(),
                "severity": check["severity"],
                "business_impact": check.get("business_impact", ""),
                "suggested_action": check.get("suggested_action", ""),
                "cost_of_ignoring": check.get("cost_of_ignoring", ""),
            })

    return {
        "audit_id": audit_id,
        "dataset_name": report["dataset_name"],
        "records_audited": report["records_audited"],
        "health_score": report["health_score"],
        "issues_summary": report["issues_summary"],
        "findings": sorted(findings, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x["severity"], 3)),
    }


@app.get("/audit/{audit_id}/fix/{check_name}")
def get_fix_code(audit_id: str, check_name: str):
    if audit_id not in audit_store:
        raise HTTPException(404, "Audit not found.")

    report = audit_store[audit_id]
    check = report["checks"].get(check_name)

    if not check:
        raise HTTPException(404, f"Check '{check_name}' not found in this audit.")

    return {
        "check": check_name,
        "fix_code": check.get("fix_code", "# No fix code available for this check."),
        "suggested_action": check.get("suggested_action", ""),
    }
