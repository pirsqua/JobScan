# JobScan

A personal pipeline that crawls company career sites directly (Greenhouse, Ashby, Lever), stores
every posting in SQLite, applies a small set of safe factual filters, and uses Claude for the
judgment-sensitive screening (product-vs-consulting, required-vs-preferred fit, compensation
credibility). It reports exactly how many companies, boards and postings it processed — it never
claims "comprehensive" coverage beyond what the employer registry actually contains.

## What this is (and isn't)

- It searches the **employer registry you give it** (`data/jobscan_remote_wlb_company_seeds_2026-09-14.yaml`
  is the current active registry — a curated list, not a claim of market coverage). Add or swap
  companies with `python -m jobscan companies import`; nothing in Python needs to change.
- It only searches for **software/backend/data engineering roles**. A crawl-time filter
  (`jobscan/job_family.py`) keeps a company's Sales, Support, Legal, People, and Product postings
  out of the database entirely — this is a job-search tool, not a generic careers-page mirror.
- Python only ever rejects **facts** (salary ceiling, explicit hybrid/on-site, explicit contract,
  a company already confirmed as a staffing shop). Every judgment call — is this really a product
  company, does "AWS or Azure" vs "deep AWS expertise" matter, is the seniority right — goes to
  Claude, once per posting, cached by description hash.

## Requirements

- Windows with PowerShell
- Python 3.12+ (developed against 3.14)
- An [Anthropic API key](https://console.anthropic.com/) for the `evaluate` step (crawling and
  reporting work without one; postings just stay "unverified" until you add a key)

## Setup (PowerShell)

```powershell
cd C:\Users\jpvin\Src\Python\JobScan

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Configure your API key
Copy-Item .env.example .env
notepad .env   # set ANTHROPIC_API_KEY=sk-ant-...
```

If PowerShell blocks the activation script, run once (as your normal user, not admin):
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Quick start (PowerShell)

```powershell
# 1. Load the curated employer registry
python -m jobscan companies import data/jobscan_remote_wlb_company_seeds_2026-09-14.yaml --replace

# 2. Crawl every active company's board
python -m jobscan crawl

# 3. Filter + screen with Claude (costs real API usage; see --limit below)
python -m jobscan evaluate

# 4. Generate Markdown/CSV/JSON reports into .\out\
python -m jobscan report

# ...or do all three in one shot:
python -m jobscan run
```

Reports land in `out\report_<timestamp>.{md,csv,json}`, timestamped in Seattle local time (e.g.
`report_20260913T223640PT.md` — the `PT` suffix covers both PST and PDT, whichever is in effect).
Open the `.md` file first — it has the run statistics, the recommended roles, and the "attractive
rejection log" (close calls worth a second look).

## Commands

| Command | What it does |
|---|---|
| `python -m jobscan companies import <file.csv\|.yaml> [--replace\|--delete]` | Add/update employers in the registry; optionally make the file the complete active set |
| `python -m jobscan companies list [--all]` | List registered companies (active only by default) |
| `python -m jobscan crawl` | Fetch current postings for every active company |
| `python -m jobscan evaluate [--limit N]` | Apply factual filters, then LLM-screen new/changed postings. `--limit` caps the number of *real LLM calls* made this run (factual filtering and cache hits are free and unaffected) — useful for bounding spend, e.g. `--limit 15` |
| `python -m jobscan report [--out DIR]` | Generate Markdown/CSV/JSON reports from the current database state |
| `python -m jobscan run [--out DIR] [--limit N]` | Crawl, evaluate, and report in sequence |
| `python -m jobscan audit [--out DIR]` | Report every filtered posting (factual or LLM) and why, so false negatives can be reviewed |
| `python -m jobscan overrides set <job\|company> <id> <field> <value> [--reason TEXT]` | Manually force a verdict (`job ... verdict strong_match`) or a company classification (`company ... classification consulting`) |

Run any command with no arguments to see this same list: `python -m jobscan --help`.

## Configuration

- **Secrets** (`ANTHROPIC_API_KEY`) go in `.env` (gitignored) or real environment variables —
  never in `config/*.yaml`.
- **`config/settings.yaml`** — the Claude model name, pricing table (for cost estimates), the
  $170,000 minimum, and the candidate's state (`WA`/`Washington`). Change the model here, not in
  code.
- **`config/candidate_profile.yaml`** — the full candidate profile (skills, target roles,
  exclusions, qualification-interpretation rules) sent to Claude on every evaluation. Edit this
  file to tune what gets recommended; no code changes needed.

## Adding more companies

```powershell
# CSV columns: name, domain, careers_url, ats_type, board_id, classification, discovery_source, notes
# ats_type is one of: greenhouse, ashby, lever
# board_id is the token in that ATS's public API URL, e.g. boards-api.greenhouse.io/v1/boards/<board_id>
python -m jobscan companies import my_companies.csv
```

YAML works too — either a bare list or `{companies: [...]}`. Field names are `board_id` and
`classification` in both CSV and YAML (no alternate spellings are accepted — keep registry files
conformed to this schema). `classification` and `discovery_source` are optional; leave
`classification` blank/`unknown` to let the LLM classify the company itself (cached indefinitely,
or override any time with `overrides set company ...`).

Import is additive by default (existing companies not in the file are left alone). Two flags
change that scope, mutually exclusive with each other:

```powershell
# The file becomes the complete active registry; anything absent is deactivated (reversible —
# history and postings are kept, and re-importing a company reactivates it).
python -m jobscan companies import my_companies.yaml --replace

# Same, but PERMANENTLY DELETES absent companies and all their job/evaluation history instead
# of deactivating them. Irreversible — only use this when you actually want that data gone.
python -m jobscan companies import my_companies.yaml --delete
```

## Manual overrides

Overrides always win over the automated pipeline and are checked before any LLM call:

```powershell
# Force a verdict on a specific job (skips both factual filters and the LLM for that job)
python -m jobscan overrides set job 123 verdict strong_match --reason "I know this team"

# Permanently mark a company as consulting/staffing (skips its postings without an LLM call)
python -m jobscan overrides set company 7 classification consulting --reason "Confirmed via LinkedIn"
```

## How it decides

**Python (facts):** salary ceiling below $170,000 (or nothing published at all), explicit
hybrid/on-site, explicit contract/part-time/intern, a company already confirmed as
consulting/staffing. A crawl-time title gate also keeps non-engineering postings out of the
database entirely (see `jobscan/job_family.py`) — this doesn't judge fit, only whether a title
reads as a software/backend/data engineering role at all.

**Claude (judgment), one call per posting, cached by description hash:** product company vs.
consulting/client-delivery, required vs. preferred qualifications, exact matches, material gaps,
minor/preferred-only gaps, whether frontend/AWS/AI-agent/distributed-systems ownership is
central, whether a $170,000+ offer is credible, and whether applying is professionally credible.
Company classification is a separate, second cached call (once per company, using its homepage
text + a sample job description), reused indefinitely.

## Data and history

Everything lives in `data\jobscan.db` (SQLite, gitignored). Closed postings are never deleted —
they're marked `status='closed'` with a `closed_at` timestamp, and every changed description is
snapshotted to `job_snapshots` before being overwritten, so history is preserved. Re-running
`crawl` is always safe: unchanged postings are untouched, changed ones are updated in place with
their prior version archived, and postings that disappeared from a board are closed (not deleted).

## Testing

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

The suite (178 tests) covers salary parsing and the $170,000 attainability rule, remote vs.
hybrid vs. explicit state exclusions, full-time vs. contract/part-time/intern,
new/changed/closed/duplicate postings, adapter pagination and partial board failures (including
real API quirks like Greenhouse sending explicit `null` for metadata/departments), manual
overrides, LLM response validation/caching, `load_settings()`'s YAML/env-var precedence, the
audit-report builder, and CLI argument parsing/dispatch — all against mocked HTTP responses and a
stub Anthropic client, so the suite never makes a real network or API call.

## Known limitations (first release)

- The employer registry is a curated list (13 companies as of 2026-09-14), not a claim of market
  coverage. The tool will always report the exact number it actually crawled, not an implied
  "comprehensive" number.
- Greenhouse/Ashby/Lever only. SmartRecruiters, Workday, and custom career sites can be added by
  implementing `jobscan.adapters.base.SourceAdapter` — the interface is designed for it, but no
  adapter exists yet.
- The job-family and location/salary parsers are regex-based heuristics tuned against real data
  from several live boards during development; they will occasionally miss an edge case (an
  unusually worded title, a novel salary format). That's what `python -m jobscan audit` and the
  manual-override commands are for.
- No web UI, scheduler, or cloud deployment — this is a CLI you run when you want fresh results.
