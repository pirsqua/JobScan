"""Command-line entry point: `python -m jobscan <command>`."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from jobscan.audit import build_audit_rows, write_audit_csv, write_audit_markdown
from jobscan.companies import RegistryImportError, import_registry
from jobscan.config import Settings, load_settings
from jobscan.crawl import crawl_all
from jobscan.db import Database
from jobscan.evaluate import evaluate_all
from jobscan.logging_setup import configure_logging, get_logger
from jobscan.models import ManualOverride
from jobscan.reports.csv_report import write_csv_report
from jobscan.reports.data import assemble_report_data
from jobscan.reports.json_report import write_json_report
from jobscan.reports.markdown_report import write_markdown_report
from jobscan.timeutil import filename_timestamp

logger = get_logger("cli")


def _open_db(settings: Settings) -> Database:
    return Database(settings.db_path)


def cmd_crawl(args: argparse.Namespace, settings: Settings) -> int:
    with _open_db(settings) as db:
        companies = db.list_companies(active_only=True)
        stats = crawl_all(db, settings, companies=companies)
    print(f"Companies attempted: {stats.companies_attempted}")
    print(f"Companies successfully crawled: {stats.companies_succeeded}")
    print(f"Failed boards: {len(stats.boards_failed)}")
    for failure in stats.boards_failed:
        print(f"  - {failure}")
    print(f"Postings fetched (all departments): {stats.postings_fetched}")
    print(f"Skipped as outside the software-engineering job family: {stats.postings_out_of_family}")
    print(f"New: {stats.new_postings}  Changed: {stats.changed_postings}  Closed: {stats.closed_postings}")
    return 0


def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    with _open_db(settings) as db:
        stats = evaluate_all(db, settings, limit=args.limit)
    print(f"Jobs considered: {stats.jobs_considered}")
    print(f"Rejected by factual filters: {stats.factual_rejected}")
    print(f"Companies newly classified: {stats.companies_classified}")
    print(f"Manual overrides applied: {stats.manual_overrides_applied}")
    print(f"Sent to LLM: {stats.sent_to_llm}  Cache hits: {stats.cache_hits}  Unverified: {stats.unverified}")
    if stats.triage_model:
        print(f"Triaged ({stats.triage_model}): {stats.triaged}  Skipped before full evaluation: {stats.triage_skipped}")
    print(f"Verdicts: {stats.verdict_counts}")
    cost = stats.estimated_cost_usd(settings)
    print(f"Tokens: {stats.input_tokens} in / {stats.output_tokens} out "
          f"(+ {stats.cache_creation_input_tokens} cache-write / {stats.cache_read_input_tokens} cache-read); "
          f"est. cost: {'$%.4f' % cost if cost is not None else 'n/a'}")
    if stats.triage_model and (stats.triage_input_tokens or stats.triage_output_tokens):
        print(f"Triage tokens: {stats.triage_input_tokens} in / {stats.triage_output_tokens} out "
              f"(+ {stats.triage_cache_creation_input_tokens} cache-write / "
              f"{stats.triage_cache_read_input_tokens} cache-read)")
    return 0


def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    out_dir = Path(args.out) if args.out else settings.output_dir
    with _open_db(settings) as db:
        crawl_stats = db.latest_crawl_run()
        eval_stats = db.latest_evaluation_run()
        data = assemble_report_data(db, settings, crawl_stats=crawl_stats, evaluate_stats=eval_stats)
    stamp = filename_timestamp()
    md_path = out_dir / f"report_{stamp}.md"
    csv_path = out_dir / f"report_{stamp}.csv"
    json_path = out_dir / f"report_{stamp}.json"
    write_markdown_report(data, md_path)
    write_csv_report(data, csv_path)
    write_json_report(data, json_path)
    print(f"Markdown: {md_path}")
    print(f"CSV:      {csv_path}")
    print(f"JSON:     {json_path}")
    return 0


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    with _open_db(settings) as db:
        companies = db.list_companies(active_only=True)
        crawl_stats = crawl_all(db, settings, companies=companies)
        eval_stats = evaluate_all(db, settings, limit=args.limit)
        data = assemble_report_data(db, settings, crawl_stats=crawl_stats, evaluate_stats=eval_stats)

    out_dir = Path(args.out) if args.out else settings.output_dir
    stamp = filename_timestamp()
    md_path = out_dir / f"report_{stamp}.md"
    csv_path = out_dir / f"report_{stamp}.csv"
    json_path = out_dir / f"report_{stamp}.json"
    write_markdown_report(data, md_path)
    write_csv_report(data, csv_path)
    write_json_report(data, json_path)

    print(f"Companies: {crawl_stats.companies_succeeded}/{crawl_stats.companies_attempted} crawled "
          f"({len(crawl_stats.boards_failed)} failed)")
    print(f"Postings: {crawl_stats.postings_fetched} fetched, {crawl_stats.postings_out_of_family} "
          f"outside the software-engineering job family (skipped), {crawl_stats.new_postings} new, "
          f"{crawl_stats.changed_postings} changed, {crawl_stats.closed_postings} closed")
    print(f"Evaluated: {eval_stats.sent_to_llm} sent to LLM, {eval_stats.cache_hits} cache hits, "
          f"{eval_stats.unverified} unverified, {eval_stats.factual_rejected} factually rejected")
    print(f"Verdicts: {eval_stats.verdict_counts}")
    print(f"Report: {md_path}")
    return 0


def cmd_companies_import(args: argparse.Namespace, settings: Settings) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1
    with _open_db(settings) as db:
        try:
            processed, total, removed = import_registry(db, path, replace=args.replace, delete=args.delete)
        except RegistryImportError as exc:
            print(f"Import failed: {exc}", file=sys.stderr)
            return 1
    print(f"Processed {processed} row(s) from {path}. Registry now has {total} companies.")
    if args.delete:
        print(f"Permanently deleted {removed} compan{'y' if removed == 1 else 'ies'} (and their job history) not present in this file.")
    elif args.replace:
        print(f"Deactivated {removed} compan{'y' if removed == 1 else 'ies'} not present in this file.")
    return 0


def cmd_companies_list(args: argparse.Namespace, settings: Settings) -> int:
    with _open_db(settings) as db:
        companies = db.list_companies(active_only=not args.all)
    for c in companies:
        crawled = c.last_crawled_at.isoformat() if c.last_crawled_at else "never"
        print(f"{c.id:>4}  {c.name:<30} {c.ats_type.value:<10} {c.board_id:<20} "
              f"{c.classification.value:<10} active={c.active} last_crawled={crawled}")
    return 0


def cmd_audit(args: argparse.Namespace, settings: Settings) -> int:
    out_dir = Path(args.out) if args.out else settings.output_dir
    with _open_db(settings) as db:
        rows = build_audit_rows(db)
    stamp = filename_timestamp()
    md_path = out_dir / f"audit_{stamp}.md"
    csv_path = out_dir / f"audit_{stamp}.csv"
    write_audit_markdown(rows, md_path)
    write_audit_csv(rows, csv_path)
    print(f"Filtered postings: {len(rows)}")
    print(f"Markdown: {md_path}")
    print(f"CSV:      {csv_path}")
    return 0


def cmd_overrides_set(args: argparse.Namespace, settings: Settings) -> int:
    with _open_db(settings) as db:
        db.set_manual_override(
            ManualOverride(
                entity_type=args.entity_type,
                entity_id=args.entity_id,
                field=args.field,
                value=args.value,
                reason=args.reason,
                created_at=datetime.now(timezone.utc),
            )
        )
    print(f"Override set: {args.entity_type}#{args.entity_id}.{args.field} = {args.value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m jobscan", description="Personal job-search discovery and evaluation pipeline.")
    parser.add_argument("--settings", default=None, help="Path to settings.yaml (default: config/settings.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_crawl = sub.add_parser("crawl", help="Fetch current postings for every active registered company.")
    p_crawl.set_defaults(func=cmd_crawl)

    p_eval = sub.add_parser("evaluate", help="Apply factual filters and run LLM screening on new/changed postings.")
    p_eval.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of real LLM calls (job + company) made this run; unlimited if omitted.",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_report = sub.add_parser("report", help="Generate Markdown/CSV/JSON reports from current database state.")
    p_report.add_argument("--out", default=None, help="Output directory (default: ./out)")
    p_report.set_defaults(func=cmd_report)

    p_run = sub.add_parser("run", help="Run crawl, evaluate and report in sequence.")
    p_run.add_argument("--out", default=None, help="Output directory (default: ./out)")
    p_run.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of real LLM calls (job + company) made this run; unlimited if omitted.",
    )
    p_run.set_defaults(func=cmd_run)

    p_audit = sub.add_parser("audit", help="Report every filtered posting and its reason.")
    p_audit.add_argument("--out", default=None, help="Output directory (default: ./out)")
    p_audit.set_defaults(func=cmd_audit)

    p_companies = sub.add_parser("companies", help="Manage the employer registry.")
    companies_sub = p_companies.add_subparsers(dest="companies_command", required=True)

    p_companies_import = companies_sub.add_parser("import", help="Import companies from a CSV or YAML file.")
    p_companies_import.add_argument("path", help="Path to a .csv or .yaml employer registry file.")
    p_companies_import_scope = p_companies_import.add_mutually_exclusive_group()
    p_companies_import_scope.add_argument(
        "--replace", action="store_true",
        help="Deactivate any registered company NOT present in this file, so it becomes the complete active registry. History is kept.",
    )
    p_companies_import_scope.add_argument(
        "--delete", action="store_true",
        help="Like --replace, but PERMANENTLY DELETES those companies and all their job/evaluation history instead of deactivating them. Irreversible.",
    )
    p_companies_import.set_defaults(func=cmd_companies_import)

    p_companies_list = companies_sub.add_parser("list", help="List registered companies.")
    p_companies_list.add_argument("--all", action="store_true", help="Include inactive companies.")
    p_companies_list.set_defaults(func=cmd_companies_list)

    p_overrides = sub.add_parser("overrides", help="Manage manual overrides.")
    overrides_sub = p_overrides.add_subparsers(dest="overrides_command", required=True)
    p_overrides_set = overrides_sub.add_parser("set", help="Set a manual override on a job or company.")
    p_overrides_set.add_argument("entity_type", choices=["job", "company"])
    p_overrides_set.add_argument("entity_id", type=int)
    p_overrides_set.add_argument("field", help="e.g. 'verdict' for a job, 'classification' for a company")
    p_overrides_set.add_argument("value")
    p_overrides_set.add_argument("--reason", default=None)
    p_overrides_set.set_defaults(func=cmd_overrides_set)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(settings_path=args.settings)
    configure_logging(settings.log_level, settings.output_dir / "logs")
    return args.func(args, settings)
