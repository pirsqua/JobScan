"""Employer registry import (CSV or YAML) — how hundreds of employers get added without
touching Python code."""
from __future__ import annotations

import csv
from pathlib import Path

import yaml

from jobscan.db import Database
from jobscan.models import AtsType, ClassificationSource, Company, CompanyClassification

REQUIRED_FIELDS = {"name", "ats_type", "board_id"}


class RegistryImportError(ValueError):
    pass


def _row_to_company(row: dict) -> Company:
    missing = REQUIRED_FIELDS - {k for k, v in row.items() if v not in (None, "")}
    if missing:
        raise RegistryImportError(f"row missing required field(s) {missing}: {row}")

    ats_raw = str(row["ats_type"]).strip().lower()
    try:
        ats_type = AtsType(ats_raw)
    except ValueError as exc:
        raise RegistryImportError(f"unknown ats_type '{ats_raw}' for company '{row.get('name')}'") from exc

    classification_raw = str(row.get("classification") or "unknown").strip().lower()
    try:
        classification = CompanyClassification(classification_raw)
    except ValueError:
        classification = CompanyClassification.UNKNOWN

    active_raw = row.get("active")
    active = True
    if active_raw is not None and str(active_raw).strip() != "":
        active = str(active_raw).strip().lower() not in ("0", "false", "no", "inactive")

    return Company(
        name=str(row["name"]).strip(),
        domain=(row.get("domain") or "").strip() or None,
        careers_url=(row.get("careers_url") or "").strip() or None,
        ats_type=ats_type,
        board_id=str(row["board_id"]).strip(),
        classification=classification,
        classification_source=ClassificationSource.SEED if classification != CompanyClassification.UNKNOWN else None,
        active=active,
        discovery_source=(row.get("discovery_source") or "").strip() or None,
        notes=(row.get("notes") or "").strip() or None,
    )


def parse_registry_file(path: Path) -> list[Company]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    elif suffix in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        rows = data.get("companies", data) if isinstance(data, dict) else data
    else:
        raise RegistryImportError(f"unsupported registry file type: {path.suffix}")

    return [_row_to_company(row) for row in rows]


def import_registry(db: Database, path: Path) -> tuple[int, int]:
    """Returns (rows_processed, companies_now_in_db)."""
    companies = parse_registry_file(path)
    for company in companies:
        db.upsert_company(company)
    total = len(db.list_companies(active_only=False))
    return len(companies), total
