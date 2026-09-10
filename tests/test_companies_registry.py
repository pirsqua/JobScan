from __future__ import annotations

from pathlib import Path

import pytest

from jobscan.companies import RegistryImportError, import_registry, parse_registry_file
from jobscan.db import Database
from jobscan.models import AtsType, CompanyClassification

CSV_CONTENT = """name,domain,careers_url,ats_type,board_id,classification,discovery_source,notes
Acme Corp,acme.example.com,https://acme.example.com/careers,greenhouse,acme,product,manual,Known good employer
Staffco,staffco.example.com,https://staffco.example.com/careers,lever,staffco,consulting,manual,Confirmed staffing agency
"""

YAML_CONTENT = """
companies:
  - name: Widgets Inc
    domain: widgets.example.com
    ats_type: ashby
    board_id: widgets
    discovery_source: manual
"""


class TestRegistryImport:
    def test_csv_import(self, tmp_path: Path, db: Database):
        path = tmp_path / "companies.csv"
        path.write_text(CSV_CONTENT, encoding="utf-8")

        processed, total = import_registry(db, path)

        assert processed == 2
        assert total == 2
        companies = {c.board_id: c for c in db.list_companies(active_only=False)}
        assert companies["acme"].ats_type == AtsType.GREENHOUSE
        assert companies["staffco"].classification == CompanyClassification.CONSULTING

    def test_yaml_import(self, tmp_path: Path, db: Database):
        path = tmp_path / "companies.yaml"
        path.write_text(YAML_CONTENT, encoding="utf-8")

        processed, total = import_registry(db, path)

        assert processed == 1
        companies = db.list_companies(active_only=False)
        assert companies[0].board_id == "widgets"
        assert companies[0].ats_type == AtsType.ASHBY

    def test_reimport_updates_rather_than_duplicates(self, tmp_path: Path, db: Database):
        path = tmp_path / "companies.csv"
        path.write_text(CSV_CONTENT, encoding="utf-8")
        import_registry(db, path)
        processed, total = import_registry(db, path)
        assert total == 2

    def test_unknown_ats_type_raises(self, tmp_path: Path):
        path = tmp_path / "bad.csv"
        path.write_text("name,ats_type,board_id\nBadCo,workday,badco\n", encoding="utf-8")
        with pytest.raises(RegistryImportError):
            parse_registry_file(path)

    def test_missing_required_field_raises(self, tmp_path: Path):
        path = tmp_path / "bad.csv"
        path.write_text("name,ats_type\nBadCo,greenhouse\n", encoding="utf-8")
        with pytest.raises(RegistryImportError):
            parse_registry_file(path)
