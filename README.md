# JobScan

[![Tests](https://github.com/pirsqua/JobScan/actions/workflows/tests.yml/badge.svg)](https://github.com/pirsqua/JobScan/actions/workflows/tests.yml)

A personal job-search pipeline. It crawls the career boards of a curated list of companies, keeps
the software-engineering postings, rejects the ones that fail hard facts (salary, remote
eligibility, employment type) in code, and has Claude judge the rest against a candidate profile.
The output is a ranked report of roles worth applying to, each linked to its posting.

**At a glance:** adapters for 16 job-board platforms · a two-model Claude pipeline — a cheap Haiku
triage, then a Sonnet evaluation returning strict structured output, with the instructions and
profile prompt-cached · code that checks the model's claims instead of trusting them · 500+
offline tests · a daily run usually costs a few cents.

It searches only the employer registry you give it, and every report states exactly how many
companies and postings were processed — never an implied "comprehensive" coverage.

## How it works

```
crawl ──► title gate ──► factual filters ──► triage (Haiku) ──► full evaluation (Sonnet) ──► report
          (code)          (code)              cheap, may skip     cached per description
```

1. **Crawl.** One adapter per job-board platform fetches every active company's current
   postings, a few boards at a time. A title gate (`jobscan/job_family.py`) drops anything that
   isn't a software, data or platform engineering role before it's stored. New, changed and
   closed postings are tracked; nothing is deleted.
2. **Factual filters (code).** A posting is rejected without any LLM call when its published
   salary range tops out below your salary floor, or it publishes no salary; when it isn't remote
   (on-site, hybrid, or never mentions remote work) or excludes your state; when it isn't
   full-time; or when the company is confirmed as consulting/staffing.
3. **Triage (cheap model).** A fast, lenient screen may skip a posting only for a short, fixed
   set of clear disqualifiers — for example a title above the level you target, a location your
   profile rules out, or a skill the posting requires that you don't have — and must quote the
   posting's own words for it. Code checks the quote is really there, decides the title-based
   reasons from the title itself, and overrules location quotes that offer remote work or don't
   actually exclude your state or time zone. Anything else goes to the full evaluation.
4. **Full evaluation (main model).** One structured call per posting, against your candidate
   profile: first what the team's engineers actually do day to day and whether that work is in
   your lane, then requirement by requirement (required vs. preferred, by the posting's own wording),
   growth dimensions, hidden staff-level scope, compensation, remote eligibility and working
   hours, then scope fit, verdict and a yes/no on whether it's worth applying. The model isn't
   deterministic, so a borderline result judged not worth applying gets one more evaluation and the
   more favourable one is kept. Companies of unknown type get one separate, cached
   product-vs-consulting call. A model that declines to answer (a safety refusal) leaves the
   posting awaiting evaluation — never rejected.
5. **Report.** Markdown, CSV and JSON in `out/`, timestamped in Pacific time.

A shortlist entry in the Markdown report looks like this (a fictional posting, judged against the
example profile):

```markdown
### Example Freight — Senior Backend Engineer, Integrations (strong_match, at_level, 86% coverage, confidence 0.80)

- Salary: $175,000 - $210,000 / year (source: structured)
- Culture: 7.5/10 (medium confidence)
- Apply: https://jobs.example.com/example-freight/senior-backend-engineer
- Freshness: published 2026-09-30; first seen 2026-10-02
- Day to day: Building and running the Python services behind carrier and customer integrations —
  REST and event-driven APIs on AWS, with an on-call rotation for them
- Exact matches: 5+ years of backend Python; REST API design and ownership; PostgreSQL;
  event-driven integrations with retries and idempotency
- Minor caveats: Kafka is preferred, where the candidate's queueing is SQS; on-call rotation
- Compensation assessment: Published base range $175,000-$210,000; the midpoint, $192,500, is above
  the $160,000 floor
- Why this is gettable: Every required qualification maps to demonstrated work — the tracking-API
  rebuild and the carrier integrations — and the scope is ordinary senior ownership on a
  structured team
```

Verdicts are cached by description text plus the details the evaluation reads beside it (title,
location, workplace type, employment type, salary): a posting is re-judged only when those change,
or on request after the rubric changes (see [Changing the rubric](#changing-the-rubric)). A role
posted twice costs one evaluation, but a copy in another location is judged on its own, since the
location decides remote eligibility.

## Design notes

The engineering choices worth a look, for anyone reading the code:

- **Field order is reasoning order.** Each LLM call returns one JSON object (structured outputs)
  with thinking off, written in schema order, so `JobEvaluationResult`
  (`jobscan/llm/schemas.py`) lists what the job is, then the evidence and analysis, then the
  verdict. With the verdict first, the model decided and then rationalized.
- **Code checks the model wherever it can.** A triage skip must quote the posting, and
  `skip_is_substantiated` (`jobscan/llm/triage.py`) verifies the quote and decides the title and
  location categories itself — it can only turn a skip into a full evaluation, never the reverse.
  The report's tiers (`jobscan/reports/data.py`) come from the requirement-by-requirement evidence
  and are capped by the fit of the day-to-day work, not the model's one-word verdict.
- **Code-side rejections err toward keeping.** No model ever sees what the title gate, the
  salary/location parsers or the factual filters reject, so each resolves ambiguity toward keeping
  a posting, and every change to them is measured against the real postings in the database.
- **Verdicts know what made them.** Each records a fingerprint of the prompt, schema, profile and
  model behind it, and every evaluation is kept. A rubric change is validated on known postings,
  sized with a dry run, applied with `--refresh-stale` — and can be reverted at no cost.
- **Built for a non-deterministic model.** The same posting can come back with different verdicts,
  so a near-miss gets a second evaluation and rubric changes are judged over several runs of known
  postings, never one.
- **Retry only what can clear up.** Every board fetch goes through one HTTP client
  (`jobscan/http_client.py`) that retries a 429, a 5xx, a timeout or a dropped connection — twice,
  with jittered backoff so parallel requests that failed together don't retry together — and
  honours a short `Retry-After`. A server asking for a long wait (a daily quota block) is left for
  the next run rather than tying up a crawl worker, and the failure is logged with what's needed to
  diagnose it. Permanent failures, like a board that moved, aren't retried at all.
- **Cost and speed are designed in.** The system prompt and profile are cached across calls, a cheap
  model screens before the expensive one, `--dry-run` prices a run before it spends, and every run
  reports its actual cost. Board fetches and LLM calls run in worker threads while every database
  read and write stays on one thread.

## Getting started (Windows, PowerShell)

You need Python 3.12+ (developed on 3.14) and, for the evaluate step, an
[Anthropic API key](https://console.anthropic.com/); crawling and reporting work without one.

**1. Install.**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dev]"
```

**2. Create your personal files in one data folder.** Three files describe you and your search:
the candidate profile, the employer registry and your applications. They live together in a data
folder that is never committed: by default the repo's gitignored `private/`, or any folder you
name in `JOBSCAN_DATA_DIR` — make that a synced folder (Google Drive, OneDrive) and they're backed
up. Their names are set in `config/settings.yaml` (`personal_files`). Start from the `.example`
versions:

```powershell
$data = "$env:USERPROFILE\OneDrive\JobScan-data"   # or "private" to keep them in the repo, unsynced
New-Item -ItemType Directory -Force $data
Copy-Item config\candidate_profile.example.yaml "$data\candidate_profile.yaml"
Copy-Item data\companies.example.yaml           "$data\jobscan_companies.yaml"
Copy-Item data\applications.example.yaml        "$data\applications.yaml"
```

Then make them yours:
- **`candidate_profile.yaml`** — your resume facts (skills, accomplishments, work history), the
  roles you want and don't, how to read qualification language, and the hard filters. It also sets
  the facts code uses: `min_base_salary`, `candidate_state` / `candidate_state_name` (the state
  you can be employed in), `candidate_time_zone`, and optionally `excluded_time_zones` (zones whose
  working hours rule a posting out, required or merely preferred). Every evaluation is judged
  against this file, so it's where to tune what gets recommended.
- **`jobscan_companies.yaml`** — the companies to search; see [The employer registry](#the-employer-registry).
- **`applications.yaml`** — delete the example entry; add one per role you apply to (see
  [Daily use](#daily-use)).

**3. Point `.env` at them.**

```powershell
Copy-Item .env.example .env
```

Set `ANTHROPIC_API_KEY`, and `JOBSCAN_DATA_DIR` to the folder from step 2: every `run` then also
ends by backing up the database there (see [Data](#data)). Leave `JOBSCAN_DATA_DIR` unset to use
the repo's `private/` — gitignored, but neither the files nor the database are backed up.

**4. Load the registry and run.**

```powershell
python -m jobscan companies import    # reads the registry in your data folder
python -m jobscan run
```

Check what the first run will cost before starting it — see [Costs](#costs). Only the `esri`
adapter renders pages in a headless browser; if you add a company on it, also run
`playwright install chromium`.

## Daily use

```powershell
python -m jobscan run        # crawl + evaluate + report, then a verified database backup
```

Open the newest `out\report_<timestamp>PT.md`. It has the run statistics, then:

| Section | Meaning |
|---|---|
| **Best Bets** | Strong match at or below your level, with every required qualification demonstrated or credibly transferable — a below-level role is flagged so you can ask about leveling |
| **Good Fits** | Plausible match at or below your level, with nothing required missing |
| **Growth Bets** | One step up (no more than one material growth dimension), or at your level but missing one required qualification |
| **Attractive Stretches** | Real overlap but a genuine stretch, still judged worth applying to — two or more required qualifications missing |
| **Rejected or Unverified** | Everything else, each with its primary reason |

The tiers are drawn from the evaluation's requirement-by-requirement evidence, not its one-word
verdict alone. Each role also says what its engineers do day to day; work partly outside your lane
caps the role at Growth Bet, and work outside it (a Node.js tool, a Go proxy) at Attractive
Stretch, however well the qualifications list reads. A role with more frontend work than your
profile accepts (say, a full-stack role that is half React, when it rules out frontend-heavy work)
is left off the shortlist.

A posting's own words on working hours are shown with the role whenever it states any; hours that
fail the working-hours filter in your profile reject it.

Record each application in your applications file (company, title, posting URL, date). The report
lists them under **Applications** with where each posting now stands — its tier, rejected, or
closed — and leaves them out of the shortlist sections, which hold only what still needs a
decision. Give an application a `status` as it moves (`screening`, `interviewing`, `offer`,
`rejected` or `withdrawn`, with an optional `status_on` date); without one it counts as awaiting a
reply. The report shows how many days each application has waited and sums up the whole funnel.

To pause a company until a date — one that accepts only a few applications per 60 days, say — list
it under `paused` in the same file with an `until` date and a reason. Its postings are still
crawled and evaluated, but stay out of the shortlist until that date, and the report's **Paused**
section says how many it is holding back.

To see why postings were filtered out — useful for spotting a wrongly rejected role — run
`python -m jobscan audit`.

## Commands

| Command | What it does |
|---|---|
| `crawl` | Fetch current postings for every active company |
| `evaluate [--limit N] [--refresh-stale] [--dry-run]` | Filter and screen new/changed postings. `--limit` caps real LLM calls; `--refresh-stale` also re-screens verdicts made under an older rubric; `--dry-run` only counts and prices what would be sent |
| `report [--out DIR]` | Write the Markdown/CSV/JSON report from the current database |
| `run [--out DIR] [--limit N] [--refresh-stale]` | `crawl`, `evaluate` and `report` in sequence, then a verified database backup into `JOBSCAN_DATA_DIR` if it is set |
| `audit [--out DIR]` | Every filtered posting and the reason (factual filter, triage or full evaluation) |
| `backup [--to DIR]` | Snapshot and verify the database now (default folder: `JOBSCAN_DATA_DIR`) — `run` already does this at its end |
| `companies import [file] [--replace\|--delete]` | Add/update companies from a YAML or CSV registry file (default: the registry in your data folder) |
| `companies list [--all]` | List companies (active only unless `--all`) |
| `companies probe <ats_type> <board_id>` | Dry-run one board through the real adapter and factual filters — what would reach evaluation, nothing stored, no LLM calls. Use it before adding a company |
| `overrides set job <id> verdict <value> [--reason TEXT]` | Force a job's verdict; bypasses filters and the LLM |
| `overrides set company <id> classification <value>` | Force a company's product/consulting classification |

All commands are `python -m jobscan <command>`; `python -m jobscan --settings PATH <command>` uses
another settings file.

## Costs

Only `evaluate` (and `run`) spend money. Crawling, reporting, auditing and cache hits are free,
and both LLM calls cache the shared system prompt and candidate profile. A daily run
sends only new or changed postings, so it usually costs cents; the first run over a new registry,
and re-screening the whole cached set after a rubric change, are the expensive operations.

Check what a run would send before running it:

```powershell
python -m jobscan evaluate --dry-run                   # what the next run would send
python -m jobscan evaluate --dry-run --refresh-stale   # what a re-screen after a rubric change would
```

A dry run applies the real filters, makes no API calls, and prints how many postings would be sent
with a cost estimate at your recent per-evaluation average. It logs no run, so the next report still
shows the last real one. Each real run prints its estimated cost, computed from the pricing table
in `config/settings.yaml` — keep that table in step with current model prices.

## Changing the rubric

The screening rubric is the system prompts (`jobscan/llm/prompts.py`), the output schemas
(`jobscan/llm/schemas.py`), your candidate profile, and the model and thinking setting that read
them (`config/settings.yaml`). Every verdict records a fingerprint of the rubric it was made
under — full evaluations and triage screen-outs separately. Changing any of them doesn't re-judge
cached verdicts; it marks them stale. `evaluate --refresh-stale` re-screens the stale ones (a
profile change marks everything stale, a triage prompt or model change only triage
screen-outs). Every evaluation is kept, so reverting a change brings the earlier verdicts back on
the next `--refresh-stale` without new API calls. The profile's code-only keys
(`candidate_state`, `candidate_time_zone`, `excluded_time_zones`) aren't part of the fingerprint.
Dry-run a refresh first (`evaluate --dry-run --refresh-stale`), and validate a change on a handful
of known postings — some that should pass, some that must stay rejected — before re-screening
everything.

## Configuration

- **`.env`** (never committed) — `ANTHROPIC_API_KEY`, your data folder (`JOBSCAN_DATA_DIR`: the
  personal files, and where `run` backs up the database), and optional overrides of
  `config/settings.yaml`: `JOBSCAN_DB_PATH`,
  `JOBSCAN_OUTPUT_DIR`, `JOBSCAN_ANTHROPIC_MODEL`, `JOBSCAN_ANTHROPIC_THINKING`,
  `JOBSCAN_TRIAGE_MODEL`, `JOBSCAN_TRIAGE_THINKING` and `JOBSCAN_LOG_LEVEL`.
- **`config/settings.yaml`** — main and triage model, pricing for cost estimates, crawl and
  evaluation concurrency (`crawl_workers`, `evaluate_workers`), HTTP settings, database and output
  paths, and the personal files' names (`personal_files`). Nothing personal.
- **Your candidate profile** — everything about you, including the salary floor, state and time
  zone; see [Getting started](#getting-started-windows-powershell).

## The employer registry

Your registry file (`jobscan_companies.yaml` in your data folder; start from
`data/companies.example.yaml`) lists the
companies to search. Code reads `name`, `domain`, `careers_url`, `ats_type`, `board_id`,
`classification`, `active`, `applied` (a label shown by `companies list`; individual applications
go in the applications file), `culture_score` and `culture_confidence`, `discovery_source` and
`notes`; the rest (`priority`, `fit_lanes`, `remote_signal`, `compensation_signal`,
`evidence_level`) is research metadata for you.

JobScan doesn't judge employer culture. Each company's `notes` carry dated evidence instead —
recent engineering-review scores, layoffs, remote policy, what its postings actually ask for. An
optional `culture_score` (0–10, your own grade of that evidence) and `culture_confidence` (how much
evidence is behind it) appear beside each of the company's roles in the report; no code filters
or ranks on them. Its `priority` grades all of it:

| Priority | Meaning |
|---|---|
| **A** | Excellent: good recent engineering culture, your stack, remote roles at your target pay, and it has produced a Best Bet |
| **A-** | Same bar, producing Growth Bets or Stretches, or posting in-stack roles regularly |
| **B+** | Good culture and in-stack remote roles reach evaluation, but nothing shortlisted yet — or a strong shortlist record with mixed culture |
| **B** | Decent, with one real question mark: thin culture evidence, infrequent in-stack roles, or mixed culture |
| **C** | A good company that rarely posts remote roles in your stack — a watch |
| **D** | Active but weak: mixed-to-poor culture or a poor fit, kept for a specific reason (you applied, or it still produces roles) |
| **F** | Inactive: the note says why — culture, no remote roles, wrong stack, or no crawlable board (kept for manual checks) |

Before adding a company, run `companies probe <ats_type> <board_id>` to see what its live postings
would send to evaluation.

`companies import` is additive. `--replace` makes the file the complete active set, deactivating
anything absent (reversible; history kept). `--delete` permanently removes absent companies and all
their job and evaluation history.

**Crawled platforms (`ats_type`):** `greenhouse`, `ashby`, `lever`, `workday`, `jobvite`,
`smartrecruiters`, `avature`, `rippling`, `jazzhr`, `icims` (classic portal), `jibe` (iCIMS career
sites), `workable`, `oracle` (Oracle Recruiting Cloud), `directemployers`, `pinpoint` and `esri`.
`board_id` is the company's token in that platform's public URL (Workday uses
`tenant/cluster/site`, and Workday, SmartRecruiters, Avature and Jibe accept the platform's own
server-side filters as a query string — see each adapter's docstring). `taleo`, `phenom`,
`eightfold`, `successfactors` and `custom` can be recorded but have no adapter; a company worth
watching on one of those is kept `active: false`, for manual checks. JobScan never works around a
deliberate anti-bot measure — boards that block automated access (Dayforce's, Eightfold tenants,
some Avature tenants) stay inactive too, to check by hand.

## Data

The database is `data/jobscan.db` (SQLite, never committed): every crawled posting and every
evaluation. Closed postings are marked closed, never deleted, and a posting's previous text is
snapshotted whenever it changes, so `crawl` is always safe to re-run. Re-judging a posting adds a
new evaluation rather than replacing the old one; the newest is current.

The database stays on the local disk — a sync client working on a live SQLite file can capture it
mid-write — and is backed up instead. When `JOBSCAN_DATA_DIR` is set, every `run` ends by
writing a consistent snapshot, running SQLite's integrity check on it, and saving it compressed as
`jobscan.db.gz` in that folder (about an eighth of the database's size), so the backup is at most
one run behind; `backup` takes one on demand. Only a snapshot that passes replaces the previous
one; a failure keeps the last good backup and prints `Database backup FAILED` (the command then
exits non-zero). To restore, decompress it over the database:

```powershell
python -c "import gzip, shutil; shutil.copyfileobj(gzip.open(r'<backup folder>\jobscan.db.gz'), open(r'data\jobscan.db', 'wb'))"
```

A synced folder such as Google Drive also keeps earlier versions of the file. Your profile,
registry and applications live wherever `.env` points.

## Development

```powershell
python -m pytest -q
ruff check . ; ruff format --check .
```

Ruff lints and formats with its default rules at a 100-character line (`pyproject.toml`). In VS
Code, the Ruff extension formats and sorts imports on save (`.vscode/settings.json`).

The tests use mocked HTTP, a stub Anthropic client and `config/candidate_profile.example.yaml` —
they never touch the network, spend money or read your personal files. Keep it that way: real
applications, posting URLs and profile details stay out of code, tests and docs. To support a new
job board, implement `jobscan.adapters.base.SourceAdapter`, register it in
`jobscan/adapters/__init__.py`, and add tests built from the board's real responses. `CLAUDE.md`
has the working conventions (cost checks, where code may and may not judge postings, schema field
order, the crawl's threading rule, where personal data lives).

## Limitations

- Coverage is exactly the registry, no more.
- The title gate and the salary/location parsers are heuristics tuned on real boards; an unusual
  title or salary format can slip through or be dropped. `audit` and manual overrides cover that.
- LLM judgment is fallible in both directions. Spot-check skips with `audit` now and then — a
  sampled second opinion on triage skips has found misses before.
- A CLI you run when you want fresh results — no scheduler or UI.

## License

MIT — see [LICENSE](LICENSE).
