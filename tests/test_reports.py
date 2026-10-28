from __future__ import annotations

from jobscan.db import Database
from jobscan.evaluate import evaluate_all
from jobscan.reports.csv_report import write_csv_report
from jobscan.reports.data import assemble_report_data
from jobscan.reports.json_report import build_json_report
from jobscan.reports.markdown_report import render_markdown

from tests.fakes import seed_company, seed_job, strong_match_client


class TestReports:
    def test_markdown_report_renders_without_evaluations(self, db: Database, settings):
        seed_job(db, settings, seed_company(db, "Acme Corp"))
        data = assemble_report_data(db, settings)
        markdown = render_markdown(data)
        assert "Run statistics" in markdown
        assert "Best Bets" in markdown

    def test_cached_verdict_is_dropped_once_the_job_fails_factual_filters(self, db: Database, settings):
        # Evaluations are cached by description text, so a re-crawl that reveals a fact the
        # verdict never saw (here: the role is actually hybrid) leaves the old strong_match in
        # place. The report must re-check the facts rather than trust the cached verdict.
        company = seed_company(db, "Acme Corp")
        seed_job(db, settings, company)
        evaluate_all(db, settings, client=strong_match_client())
        assert assemble_report_data(db, settings).best_bets

        seed_job(db, settings, company, location="Hybrid - Seattle, WA")
        assert assemble_report_data(db, settings).best_bets == []

    def test_markdown_report_lists_recommendation_after_evaluation(self, db: Database, settings):
        seed_job(db, settings, seed_company(db, "Acme Corp"))
        eval_stats = evaluate_all(db, settings, client=strong_match_client())

        data = assemble_report_data(db, settings, evaluate_stats=eval_stats)
        markdown = render_markdown(data)
        assert "Acme Corp" in markdown
        assert "strong_match" in markdown

        json_report = build_json_report(data)
        assert json_report["best_bets"][0]["company"] == "Acme Corp"

        csv_path = settings.output_dir / "report.csv"
        write_csv_report(data, csv_path)
        assert csv_path.exists()
        assert "Acme Corp" in csv_path.read_text(encoding="utf-8")

    def test_culture_score_is_shown_beside_each_role_when_the_registry_has_one(self, db: Database, settings):
        company = seed_company(db, "Acme Corp")
        seed_job(db, settings, company)
        evaluate_all(db, settings, client=strong_match_client())
        assert "- Culture:" not in render_markdown(assemble_report_data(db, settings))

        company.culture_score, company.culture_confidence = 7.5, "medium"
        db.upsert_company(company)
        data = assemble_report_data(db, settings)
        assert "- Culture: 7.5/10 (medium confidence)" in render_markdown(data)
        best = build_json_report(data)["best_bets"][0]
        assert (best["company_culture_score"], best["company_culture_confidence"]) == (7.5, "medium")
        csv_path = settings.output_dir / "report.csv"
        write_csv_report(data, csv_path)
        header, first = csv_path.read_text(encoding="utf-8").splitlines()[:2]
        assert header.startswith("company,culture_score,culture_confidence,") and first.startswith("Acme Corp,7.5,medium,")

    def test_pool_is_partitioned_and_kept_apart_from_the_latest_run(self, db: Database, settings):
        company = seed_company(db, "Acme Corp")
        seed_job(db, settings, company)
        seed_job(db, settings, company, "2", "Backend Engineer", "No salary here.")
        evaluate_all(db, settings, client=strong_match_client())
        seed_job(db, settings, company, "3", description="$190,000 - $230,000 per year.")
        second = evaluate_all(db, settings)  # no key: the new posting stays awaiting evaluation

        data = assemble_report_data(db, settings, evaluate_stats=second)

        pool = data.pool
        assert (pool.active_postings, pool.factually_rejected, pool.evaluated, pool.awaiting_evaluation) == (3, 1, 1, 1)
        assert pool.verdict_counts == {"strong_match": 1}
        markdown = render_markdown(data)
        latest = markdown[markdown.index("### Latest run"):markdown.index("### Current pool")]
        assert "**0** newly rejected by factual filters" in latest
        assert "New verdicts: none" in latest
        current = markdown[markdown.index("### Current pool"):]
        assert "Rejected by factual filters (salary, remote, employment type, consulting): **1**" in current
        assert "Awaiting evaluation: **1**" in current
        assert build_json_report(data)["pool"]["evaluated"] == 1

    def test_a_deactivated_companys_postings_leave_the_report(self, db: Database, settings):
        company = seed_company(db, "Acme Corp")
        seed_job(db, settings, company)
        evaluate_all(db, settings, client=strong_match_client())
        db.conn.execute("UPDATE companies SET active = 0 WHERE id = ?", (company.id,))

        data = assemble_report_data(db, settings)

        assert data.best_bets == []
        assert data.pool.active_postings == 0
