"""Command-line entry point: `python -m jobscan <command>`."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from jobscan.adapters import ADAPTERS
from jobscan.adapters.base import AdapterError
from jobscan.audit import build_audit_rows, write_audit_csv, write_audit_markdown
from jobscan.backup import BackupError, backup_database
from jobscan.companies import RegistryImportError, import_registry
from jobscan.config import Settings, load_settings
from jobscan.crawl import crawl_all
from jobscan.db import Database
from jobscan.evaluate import evaluate_all
from jobscan.http_client import make_client
from jobscan.logging_setup import configure_logging, get_logger
from jobscan.models import AtsType, EvaluateStats, ManualOverride
from jobscan.probe import probe_board
from jobscan.reports.csv_report import write_csv_report
from jobscan.reports.data import ReportData, assemble_report_data
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
    print(
        f"New: {stats.new_postings}  Changed: {stats.changed_postings}  Closed: {stats.closed_postings}"
    )
    return 0


def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    with _open_db(settings) as db:
        stats = evaluate_all(
            db, settings, limit=args.limit, refresh_stale=args.refresh_stale, dry_run=args.dry_run
        )
        average = db.average_evaluation_tokens(settings.anthropic_model) if args.dry_run else None
    if args.dry_run:
        print(f"Dry run, no API calls: {stats.unverified} postings would be sent to the LLM")
        if average is not None and stats.unverified:
            per_posting = average.estimated_cost_usd(settings)
            if per_posting is not None:
                print(
                    f"At ${per_posting:.3f} per full evaluation (recent average), about "
                    f"${per_posting * stats.unverified:.2f} if every one reaches it — triage skips "
                    "some, and near-misses get a second look"
                )
        return 0
    _print_evaluation_run(stats, settings)
    print(
        f"Pool: {stats.jobs_considered} active postings, {stats.factual_rejected} rejected by factual filters, "
        f"{sum(stats.verdict_counts.values())} evaluated, {stats.unverified} awaiting evaluation"
    )
    return 0


def _print_evaluation_run(stats: EvaluateStats, settings: Settings) -> None:
    """This run's own work only — the pool-wide counts are printed separately."""
    triage = (
        f"{stats.triaged} triaged ({stats.triage_skipped} skipped, {stats.triage_overruled} skips overruled), "
        if stats.triage_model
        else ""
    )
    second = (
        f", {stats.second_looks} second looks ({stats.second_looks_kept} better)"
        if stats.second_looks
        else ""
    )
    cost = stats.estimated_cost_usd(settings)
    print(
        f"This run: {stats.newly_filtered} newly rejected by factual filters, {triage}"
        f"{stats.sent_to_llm} full evaluations{second}, {stats.llm_errors} LLM errors; "
        f"cost {f'${cost:.2f}' if cost is not None else 'n/a'}"
    )
    print(f"New verdicts: {stats.new_verdict_counts or 'none'}")


def _write_reports(data: ReportData, out_dir: Path) -> tuple[Path, Path, Path]:
    """The Markdown, CSV and JSON reports for one snapshot, sharing a Seattle-time stamp."""
    stamp = filename_timestamp()
    paths = (
        out_dir / f"report_{stamp}.md",
        out_dir / f"report_{stamp}.csv",
        out_dir / f"report_{stamp}.json",
    )
    write_markdown_report(data, paths[0])
    write_csv_report(data, paths[1])
    write_json_report(data, paths[2])
    return paths


def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    out_dir = Path(args.out) if args.out else settings.output_dir
    with _open_db(settings) as db:
        crawl_stats = db.latest_crawl_run()
        eval_stats = db.latest_evaluation_run()
        data = assemble_report_data(
            db, settings, crawl_stats=crawl_stats, evaluate_stats=eval_stats
        )
    md_path, csv_path, json_path = _write_reports(data, out_dir)
    print(f"Markdown: {md_path}")
    print(f"CSV:      {csv_path}")
    print(f"JSON:     {json_path}")
    return 0


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    with _open_db(settings) as db:
        companies = db.list_companies(active_only=True)
        crawl_stats = crawl_all(db, settings, companies=companies)
        eval_stats = evaluate_all(db, settings, limit=args.limit, refresh_stale=args.refresh_stale)
        data = assemble_report_data(
            db, settings, crawl_stats=crawl_stats, evaluate_stats=eval_stats
        )

    out_dir = Path(args.out) if args.out else settings.output_dir
    md_path, _, _ = _write_reports(data, out_dir)

    print(
        f"Crawl: {crawl_stats.companies_succeeded}/{crawl_stats.companies_attempted} companies "
        f"({len(crawl_stats.boards_failed)} failed); {crawl_stats.new_postings} new, "
        f"{crawl_stats.changed_postings} changed, {crawl_stats.closed_postings} closed postings"
    )
    _print_evaluation_run(eval_stats, settings)
    pool = data.pool
    print(
        f"Pool: {pool.active_postings} active postings — {pool.factually_rejected} rejected by factual filters, "
        f"{pool.evaluated} evaluated, {pool.awaiting_evaluation} awaiting evaluation"
    )
    print(
        f"Shortlist (not yet applied to): {len(data.best_bets)} Best Bets, {len(data.good_fits)} Good Fits, "
        f"{len(data.growth_bets)} Growth Bets, {len(data.attractive_stretches)} Attractive Stretches"
    )
    print(f"Report: {md_path}")
    # The daily command, so its end is where the database gets backed up.
    if settings.backup_dir is not None:
        return _report_backup(settings, settings.backup_dir)
    return 0


def _report_backup(settings: Settings, target: Path) -> int:
    """Snapshot and verify the database; printed to stdout either way, so a failure is never
    hidden behind a silenced stderr."""
    try:
        snapshot = backup_database(settings.db_path, target)
    except BackupError as exc:
        print(f"Database backup FAILED: {exc}")
        return 1
    print(
        f"Database backup: {snapshot} ({snapshot.stat().st_size / 1e6:.1f} MB compressed, integrity check passed)"
    )
    return 0


def cmd_backup(args: argparse.Namespace, settings: Settings) -> int:
    target = Path(args.to) if args.to else settings.backup_dir
    if target is None:
        print("No backup folder given and JOBSCAN_DATA_DIR is not set.", file=sys.stderr)
        return 1
    return _report_backup(settings, target)


def cmd_companies_import(args: argparse.Namespace, settings: Settings) -> int:
    if args.path is None and settings.registry_path is None:
        print("No registry file given.", file=sys.stderr)
        return 1
    path = Path(args.path) if args.path else settings.registry_path
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1
    with _open_db(settings) as db:
        try:
            processed, total, removed = import_registry(
                db, path, replace=args.replace, delete=args.delete
            )
        except RegistryImportError as exc:
            print(f"Import failed: {exc}", file=sys.stderr)
            return 1
    print(f"Processed {processed} row(s) from {path}. Registry now has {total} companies.")
    if args.delete:
        print(
            f"Permanently deleted {removed} compan{'y' if removed == 1 else 'ies'} (and their job history) not present in this file."
        )
    elif args.replace:
        print(
            f"Deactivated {removed} compan{'y' if removed == 1 else 'ies'} not present in this file."
        )
    return 0


def cmd_companies_list(args: argparse.Namespace, settings: Settings) -> int:
    with _open_db(settings) as db:
        companies = db.list_companies(active_only=not args.all)
    for c in companies:
        crawled = c.last_crawled_at.isoformat() if c.last_crawled_at else "never"
        applied_tag = " [APPLIED]" if c.applied else ""
        print(
            f"{c.id:>4}  {c.name:<30} {c.ats_type.value:<10} {c.board_id:<20} "
            f"{c.classification.value:<10} active={c.active} last_crawled={crawled}{applied_tag}"
        )
    return 0


def cmd_companies_probe(args: argparse.Namespace, settings: Settings) -> int:
    try:
        ats_type = AtsType(args.ats_type)
    except ValueError:
        print(f"Unknown ats_type {args.ats_type!r}", file=sys.stderr)
        return 1
    if ats_type not in ADAPTERS:
        print(
            f"No adapter for {ats_type.value} — such a company can only be recorded inactive, for manual checks.",
            file=sys.stderr,
        )
        return 1
    with make_client(settings) as client:
        try:
            result = probe_board(settings, ats_type, args.board_id, client)
        except AdapterError as exc:
            print(f"Board fetch failed: {exc}", file=sys.stderr)
            return 1
    filtered = (
        ", ".join(f"{reason} {count}" for reason, count in result.rejected.most_common()) or "none"
    )
    print(
        f"{ats_type.value}/{args.board_id}: {result.postings} postings, {result.engineering} engineering, "
        f"{len(result.passing)} would reach evaluation (filtered: {filtered})"
    )
    for job in result.passing:
        pay = (
            f"${job.salary_min:,.0f}-${(job.salary_max or job.salary_min):,.0f}"
            if job.salary_min
            else "pay n/a"
        )
        print(f"  {job.title} | {job.location_raw or '-'} | {pay} | {job.posting_url or '-'}")
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
                created_at=datetime.now(UTC),
            )
        )
    print(f"Override set: {args.entity_type}#{args.entity_id}.{args.field} = {args.value}")
    return 0


_REFRESH_STALE_HELP = (
    "Re-screen postings whose cached verdict was made under an older rubric (prompts or candidate "
    "profile changed since). Spends money — dry-run first; combine with --limit to cap it."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jobscan",
        description="Personal job-search discovery and evaluation pipeline.",
    )
    parser.add_argument(
        "--settings", default=None, help="Path to settings.yaml (default: config/settings.yaml)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_crawl = sub.add_parser(
        "crawl", help="Fetch current postings for every active registered company."
    )
    p_crawl.set_defaults(func=cmd_crawl)

    p_eval = sub.add_parser(
        "evaluate", help="Apply factual filters and run LLM screening on new/changed postings."
    )
    p_eval.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of real LLM calls (job + company) made this run; unlimited if omitted.",
    )
    p_eval.add_argument("--refresh-stale", action="store_true", help=_REFRESH_STALE_HELP)
    p_eval.add_argument(
        "--dry-run",
        action="store_true",
        help="Make no API calls: count what would be sent and estimate the cost. Factual filtering "
        "and free cache reuse still happen.",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_report = sub.add_parser(
        "report", help="Generate Markdown/CSV/JSON reports from current database state."
    )
    p_report.add_argument("--out", default=None, help="Output directory (default: ./out)")
    p_report.set_defaults(func=cmd_report)

    p_run = sub.add_parser("run", help="Run crawl, evaluate and report in sequence.")
    p_run.add_argument("--out", default=None, help="Output directory (default: ./out)")
    p_run.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of real LLM calls (job + company) made this run; unlimited if omitted.",
    )
    p_run.add_argument("--refresh-stale", action="store_true", help=_REFRESH_STALE_HELP)
    p_run.set_defaults(func=cmd_run)

    p_backup = sub.add_parser(
        "backup", help="Snapshot the database to a backup folder (default: JOBSCAN_DATA_DIR)."
    )
    p_backup.add_argument("--to", default=None, help="Backup folder (default: JOBSCAN_DATA_DIR)")
    p_backup.set_defaults(func=cmd_backup)

    p_audit = sub.add_parser("audit", help="Report every filtered posting and its reason.")
    p_audit.add_argument("--out", default=None, help="Output directory (default: ./out)")
    p_audit.set_defaults(func=cmd_audit)

    p_companies = sub.add_parser("companies", help="Manage the employer registry.")
    companies_sub = p_companies.add_subparsers(dest="companies_command", required=True)

    p_companies_import = companies_sub.add_parser(
        "import", help="Import companies from a CSV or YAML file."
    )
    p_companies_import.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to a .csv or .yaml employer registry file (default: the registry in JOBSCAN_DATA_DIR).",
    )
    p_companies_import_scope = p_companies_import.add_mutually_exclusive_group()
    p_companies_import_scope.add_argument(
        "--replace",
        action="store_true",
        help="Deactivate any registered company NOT present in this file, so it becomes the complete active registry. History is kept.",
    )
    p_companies_import_scope.add_argument(
        "--delete",
        action="store_true",
        help="Like --replace, but PERMANENTLY DELETES those companies and all their job/evaluation history instead of deactivating them. Irreversible.",
    )
    p_companies_import.set_defaults(func=cmd_companies_import)

    p_companies_list = companies_sub.add_parser("list", help="List registered companies.")
    p_companies_list.add_argument("--all", action="store_true", help="Include inactive companies.")
    p_companies_list.set_defaults(func=cmd_companies_list)

    p_companies_probe = companies_sub.add_parser(
        "probe",
        help="Dry-run one board through the real adapter and factual filters — nothing stored, no LLM calls.",
    )
    p_companies_probe.add_argument("ats_type", help="e.g. greenhouse, ashby, lever, workday")
    p_companies_probe.add_argument(
        "board_id", help="The board's id, exactly as a registry row would give it"
    )
    p_companies_probe.set_defaults(func=cmd_companies_probe)

    p_overrides = sub.add_parser("overrides", help="Manage manual overrides.")
    overrides_sub = p_overrides.add_subparsers(dest="overrides_command", required=True)
    p_overrides_set = overrides_sub.add_parser(
        "set", help="Set a manual override on a job or company."
    )
    p_overrides_set.add_argument("entity_type", choices=["job", "company"])
    p_overrides_set.add_argument("entity_id", type=int)
    p_overrides_set.add_argument(
        "field", help="e.g. 'verdict' for a job, 'classification' for a company"
    )
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
