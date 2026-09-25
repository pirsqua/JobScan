from __future__ import annotations

from pathlib import Path

import pytest

from jobscan.companies import RegistryImportError, import_registry, parse_registry_file
from jobscan.db import Database
from jobscan.models import AtsType, CompanyClassification, RawPosting
from jobscan.normalize import normalize_posting

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

        processed, total, deactivated = import_registry(db, path)

        assert processed == 2
        assert total == 2
        assert deactivated == 0
        companies = {c.board_id: c for c in db.list_companies(active_only=False)}
        assert companies["acme"].ats_type == AtsType.GREENHOUSE
        assert companies["staffco"].classification == CompanyClassification.CONSULTING

    def test_yaml_import(self, tmp_path: Path, db: Database):
        path = tmp_path / "companies.yaml"
        path.write_text(YAML_CONTENT, encoding="utf-8")

        processed, total, _ = import_registry(db, path)

        assert processed == 1
        companies = db.list_companies(active_only=False)
        assert companies[0].board_id == "widgets"
        assert companies[0].ats_type == AtsType.ASHBY

    def test_reimport_updates_rather_than_duplicates(self, tmp_path: Path, db: Database):
        path = tmp_path / "companies.csv"
        path.write_text(CSV_CONTENT, encoding="utf-8")
        import_registry(db, path)
        _, total, _ = import_registry(db, path)
        assert total == 2

    def test_replace_deactivates_companies_not_in_new_file(self, tmp_path: Path, db: Database):
        old_path = tmp_path / "old.csv"
        old_path.write_text(CSV_CONTENT, encoding="utf-8")
        import_registry(db, old_path)

        new_path = tmp_path / "new.yaml"
        new_path.write_text(YAML_CONTENT, encoding="utf-8")
        processed, total, deactivated = import_registry(db, new_path, replace=True)

        assert processed == 1
        assert deactivated == 2  # Acme Corp and Staffco, both absent from the new file
        active_names = {c.name for c in db.list_companies(active_only=True)}
        assert active_names == {"Widgets Inc"}
        assert len(db.list_companies(active_only=False)) == 3  # history preserved, not deleted

    def test_replace_keeps_company_present_in_both_files(self, tmp_path: Path, db: Database):
        old_path = tmp_path / "old.csv"
        old_path.write_text(CSV_CONTENT, encoding="utf-8")
        import_registry(db, old_path)

        # Re-import just Acme via replace=True; Staffco should be deactivated, Acme kept active.
        new_path = tmp_path / "new.csv"
        new_path.write_text(
            "name,domain,ats_type,board_id,classification\n"
            "Acme Corp,acme.example.com,greenhouse,acme,product\n",
            encoding="utf-8",
        )
        _, _, deactivated = import_registry(db, new_path, replace=True)

        assert deactivated == 1
        active_boards = {c.board_id for c in db.list_companies(active_only=True)}
        assert active_boards == {"acme"}

    def test_delete_permanently_removes_companies_and_their_jobs(self, tmp_path: Path, db: Database, settings):
        old_path = tmp_path / "old.csv"
        old_path.write_text(CSV_CONTENT, encoding="utf-8")
        import_registry(db, old_path)
        acme = next(c for c in db.list_companies(active_only=False) if c.board_id == "acme")

        raw = RawPosting(
            source_job_id="1", title="Senior Backend Engineer", location_raw="Remote - US",
            employment_type_raw="Full-time", description_html="<p>$180,000 - $220,000</p>",
            description_text=None, posting_url="https://acme.example.com/1", apply_url=None,
        )
        job = normalize_posting(raw, acme, settings)
        db.upsert_job(job)
        assert db.count_jobs() == 1

        new_path = tmp_path / "new.yaml"
        new_path.write_text(YAML_CONTENT, encoding="utf-8")
        processed, total, removed = import_registry(db, new_path, delete=True)

        assert processed == 1
        assert removed == 2  # Acme Corp and Staffco
        assert total == 1
        assert db.count_jobs() == 0  # Acme's job history was removed along with the company
        assert {c.name for c in db.list_companies(active_only=False)} == {"Widgets Inc"}

    def test_unknown_ats_type_raises(self, tmp_path: Path):
        path = tmp_path / "bad.csv"
        path.write_text("name,ats_type,board_id\nBadCo,bamboohr,badco\n", encoding="utf-8")
        with pytest.raises(RegistryImportError):
            parse_registry_file(path)

    def test_recorded_unsupported_ats_types_do_not_raise(self, tmp_path: Path, db: Database):
        # These have no crawl adapter yet, but a registry curator should be able to record that a
        # company uses one (typically active=false) without the import failing — see AtsType's
        # docstring. (Greenhouse/Ashby/Lever/Workday/Jobvite/Esri are the supported ones and are
        # exercised elsewhere, not here.)
        path = tmp_path / "unsupported.csv"
        path.write_text(
            "name,ats_type,board_id,active\n"
            "SmartRecruiters Co,smartrecruiters,smartco,false\n"
            "iCIMS Co,icims,icimsco,false\n"
            "Taleo Co,taleo,taleoco,false\n"
            "Avature Co,avature,avatureco,false\n"
            "Phenom Co,phenom,phenomco,false\n"
            "Eightfold Co,eightfold,eightfoldco,false\n"
            "Custom Co,custom,customco,false\n",
            encoding="utf-8",
        )
        processed, total, _ = import_registry(db, path)
        assert processed == 7
        assert total == 7
        boards = {c.board_id: c for c in db.list_companies(active_only=False)}
        assert boards["smartco"].ats_type == AtsType.SMARTRECRUITERS
        assert boards["smartco"].active is False
        assert boards["icimsco"].ats_type == AtsType.ICIMS
        assert boards["taleoco"].ats_type == AtsType.TALEO
        assert boards["avatureco"].ats_type == AtsType.AVATURE
        assert boards["phenomco"].ats_type == AtsType.PHENOM
        assert boards["eightfoldco"].ats_type == AtsType.EIGHTFOLD

    def test_missing_required_field_raises(self, tmp_path: Path):
        path = tmp_path / "bad.csv"
        path.write_text("name,ats_type\nBadCo,greenhouse\n", encoding="utf-8")
        with pytest.raises(RegistryImportError):
            parse_registry_file(path)

    def test_blank_board_id_falls_back_to_domain(self, tmp_path: Path, db: Database):
        # "custom" (bespoke career-page) entries have no board-token API, so no real board_id —
        # domain is a stable, unique-per-company fallback rather than requiring one be invented.
        path = tmp_path / "custom.csv"
        path.write_text(
            "name,domain,ats_type,board_id,active\nBitwarden,bitwarden.com,custom,,false\n",
            encoding="utf-8",
        )
        companies = parse_registry_file(path)
        assert companies[0].board_id == "bitwarden.com"

    def test_blank_board_id_without_domain_still_raises(self, tmp_path: Path):
        path = tmp_path / "bad.csv"
        path.write_text("name,ats_type,board_id,active\nBadCo,custom,,false\n", encoding="utf-8")
        with pytest.raises(RegistryImportError):
            parse_registry_file(path)
