# JobScan

A personal job-search pipeline. It crawls the career boards of a curated list of companies, keeps
the software-engineering postings, rejects the ones that fail hard facts (salary, remote
eligibility, employment type) in code, and has Claude judge the rest against a candidate profile.
The output is a ranked report of roles worth applying to, each linked to its posting.

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
   salary range tops out below `min_base_salary`, or it publishes no salary; when it isn't remote
   (on-site, hybrid, or never mentions remote work) or excludes the candidate's state; when it
   isn't full-time; or when the company is confirmed as consulting/staffing.
3. **Triage (cheap model).** A fast, lenient screen may skip a posting only for a fixed list of
   reasons — Staff-or-above title, people management, not remote, excluded industry, not an
   engineering job, a required unfamiliar language, a required specialty — and must quote the
   posting's own words for it. Code checks the quote is really there, decides Staff+ from the title
   (unless the posting says it hires at Senior too) and management from a manager title, and
   overrules location quotes that offer remote work or don't actually exclude the candidate's state
   or time zone. Anything else goes to the full evaluation.
4. **Full evaluation (main model).** One structured call per posting, against
   `config/candidate_profile.yaml`: requirement by requirement (required vs. preferred, by the
   posting's own wording), growth dimensions, hidden staff-level scope, compensation, remote
   eligibility and working hours, then scope fit, verdict and a yes/no on whether it's worth
   applying. The model isn't deterministic, so a borderline result judged not worth applying gets
   one more evaluation and the more favourable one is kept. Companies of unknown type get one
   separate, cached product-vs-consulting call.
5. **Report.** Markdown, CSV and JSON in `out/`, timestamped in Seattle time.

Verdicts are cached by description text: a posting is re-judged only when its text changes, or on
request after the rubric changes (see [Changing the rubric](#changing-the-rubric)).

## Setup (Windows, PowerShell)

Needs Python 3.12+ (developed on 3.14) and an [Anthropic API key](https://console.anthropic.com/)
for the evaluate step; crawling and reporting work without one.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dev]"
Copy-Item .env.example .env           # then set ANTHROPIC_API_KEY=sk-ant-...
python -m jobscan companies import data/jobscan_companies_2026-09-29.yaml
```

Only the `esri` adapter renders pages in a headless browser, and no company in the registry uses
it; if you add one, also run `playwright install chromium`.

## Daily use

```powershell
python -m jobscan run        # crawl + evaluate + report
```

Open the newest `out\report_<timestamp>PT.md`. It has the run statistics, then:

| Section | Meaning |
|---|---|
| **Best Bets** | Strong or plausible match at your level |
| **Growth Bets** | Plausible match one step up — no more than one material growth dimension |
| **Attractive Stretches** | Real overlap but a genuine stretch, still judged worth applying to — or a good fit below your level |
| **Rejected or Unverified** | Everything else, each with its primary reason |

A posting's own words on working hours are shown with the role whenever it states any; Eastern or
Central hours, required or preferred, reject it.

Record each application in `data/applications.yaml` (company, title, posting URL, date). The report
lists them under **Applications** with where each posting now stands — still shortlisted, rejected,
or closed — and stars (★) every applied role in its section.

To see why postings were filtered out — useful for spotting a wrongly rejected role — run
`python -m jobscan audit`.

## Commands

| Command | What it does |
|---|---|
| `crawl` | Fetch current postings for every active company |
| `evaluate [--limit N] [--refresh-stale]` | Filter and screen new/changed postings. `--limit` caps real LLM calls; `--refresh-stale` also re-screens verdicts made under an older rubric |
| `report [--out DIR]` | Write the Markdown/CSV/JSON report from the current database |
| `run [--out DIR] [--limit N] [--refresh-stale]` | `crawl`, `evaluate` and `report` in sequence |
| `audit [--out DIR]` | Every filtered posting and the reason (factual filter, triage or full evaluation) |
| `companies import <file> [--replace\|--delete]` | Add/update companies from a YAML or CSV registry file |
| `companies list [--all]` | List companies (active only unless `--all`) |
| `companies probe <ats_type> <board_id>` | Dry-run one board through the real adapter and factual filters — what would reach evaluation, nothing stored, no LLM calls. Use it before adding a company |
| `overrides set job <id> verdict <value> [--reason TEXT]` | Force a job's verdict; bypasses filters and the LLM |
| `overrides set company <id> classification <value>` | Force a company's product/consulting classification |

All commands are `python -m jobscan <command>`; `python -m jobscan --settings PATH <command>` uses
another settings file.

## Costs

Only `evaluate` (and `run`) spend money. Crawling, reporting, auditing and cache hits are free,
and both LLM calls cache the shared system prompt, tool schema and candidate profile. A daily run
sends only new or changed postings, so it usually costs cents; re-screening the whole cached set
after a rubric change is the expensive operation.

Check what a run would send before running it — this applies the real filters with no API calls
and stores no verdicts (it does log a run, so a `report` right after it shows the dry run's
statistics until the next real `evaluate`):

```powershell
python -c "import dataclasses; from jobscan.config import load_settings; from jobscan.db import Database; from jobscan.evaluate import evaluate_all; s = dataclasses.replace(load_settings(), anthropic_api_key=None); print(evaluate_all(Database(s.db_path), s).unverified, 'postings would be sent')"
```

(`evaluate_all(..., refresh_stale=True)` counts a full re-screen the same way.) Each run prints its
estimated cost, computed from the pricing table in `config/settings.yaml` — keep that table in
step with current model prices.

## Changing the rubric

The screening rubric is the system prompts (`jobscan/llm/prompts.py`), the tool schemas
(`jobscan/llm/schemas.py`) and `config/candidate_profile.yaml`. Every verdict records a
fingerprint of the rubric it was made under — full evaluations and triage screen-outs separately.
Editing any of them doesn't re-judge cached verdicts; it marks them stale.
`evaluate --refresh-stale` re-screens the stale ones (a profile change marks everything stale, a
triage-prompt change only triage screen-outs). Dry-run it first, and validate a change on a handful
of known postings — some that should pass, some that must stay rejected — before re-screening
everything.

## Configuration

- **`.env`** — `ANTHROPIC_API_KEY` (never in a YAML file).
- **`config/settings.yaml`** — main and triage model, pricing for cost estimates,
  `min_base_salary`, the candidate's state and time zone, crawl concurrency (`crawl_workers`) and
  HTTP settings, database and output paths.
- **`config/candidate_profile.yaml`** — skills, accomplishments, target roles, roles to avoid,
  qualification-interpretation rules, the compensation rule and hard filters, sent with every
  evaluation. This is where to tune what gets recommended.

## The employer registry

`data/jobscan_companies_2026-09-29.yaml` is the list of companies to search. Code reads `name`,
`domain`, `careers_url`, `ats_type`, `board_id`, `classification`, `active`, `applied` (a label
shown by `companies list`; individual applications go in `data/applications.yaml`),
`discovery_source` and `notes`; the rest (`priority`, `fit_lanes`, `remote_signal`,
`compensation_signal`, `evidence_level`) is research metadata for you.

JobScan doesn't judge employer culture. Each company's `notes` carry dated evidence instead —
recent engineering-review scores, layoffs, remote policy, what its postings actually ask for — and
its `priority` grades all of it:

| Priority | Meaning |
|---|---|
| **A** | Excellent: good recent engineering culture, your stack, remote-US roles at the target pay, and it has produced a Best Bet |
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
`smartrecruiters`, `avature`, `rippling`, `jazzhr`, `icims` (classic portal), `jibe` (iCIMS career sites),
`workable`, `oracle` (Oracle Recruiting Cloud), `directemployers`, `pinpoint` and `esri`. `board_id`
is the company's token in that platform's public URL (Workday uses `tenant/cluster/site`, and
Workday, SmartRecruiters, Avature and Jibe accept the platform's own server-side filters as a query
string — see each adapter's docstring). `taleo`, `phenom`, `eightfold`, `successfactors` and
`custom` can be recorded but have no adapter; a company worth watching on one of those is kept
`active: false`, for manual checks. JobScan never works around a deliberate anti-bot measure —
boards that block automated access (Dayforce's, Eightfold tenants, some Avature tenants) stay
inactive too, to check by hand.

## Data

Everything lives in `data/jobscan.db` (SQLite, not committed). Closed postings are marked closed,
never deleted, and a posting's previous text is snapshotted whenever it changes, so `crawl` is
always safe to re-run.

## Development

```powershell
python -m pytest -q
```

The tests use mocked HTTP and a stub Anthropic client — they never touch the network or spend
money. To support a new job board, implement `jobscan.adapters.base.SourceAdapter`, register it in
`jobscan/adapters/__init__.py`, and add tests built from the board's real responses.
`CLAUDE.md` has the working conventions (cost checks, where code may and may not judge postings,
schema field order, the crawl's threading rule).

## Limitations

- Coverage is exactly the registry, no more.
- The title gate and the salary/location parsers are heuristics tuned on real boards; an unusual
  title or salary format can slip through or be dropped. `audit` and manual overrides cover that.
- LLM judgment is fallible in both directions. Spot-check skips with `audit` now and then — a
  sampled second opinion on triage skips has found misses before.
- A CLI you run when you want fresh results — no scheduler or UI.
