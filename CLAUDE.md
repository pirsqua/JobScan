# Operating notes for Claude Code in this repo

For what the project *is* and how to run it, read `README.md` — this file is only for things a
fresh session needs before touching code, that reading the code alone won't make obvious.

## Environment

- Windows, PowerShell primary; venv at `.venv` (`.\.venv\Scripts\Activate.ps1`).
- Python 3.14. Run the suite before considering any change done: `python -m pytest -q`.
- The owner's personal data — candidate profile, applications, employer registry — lives in one
  data folder outside the repository, the one `.env` sets as `JOBSCAN_DATA_DIR` (file names in
  `config/settings.yaml`, `personal_files`); edit it there. Never commit it, never copy it into the repo, and never
  put a real application, posting URL or profile detail into code, tests or docs — use the
  `.example` files and example.com URLs. Tests read `config/candidate_profile.example.yaml`.
- Commit messages carry no `Co-Authored-By` or "Generated with" trailers.

## Cost awareness — `evaluate` and `run` spend real money

They call the real Anthropic API (model/pricing in `config/settings.yaml`). Before running either
without `--limit` against the full registry:
1. Do a free dry run first — `python -m jobscan evaluate --dry-run` (with `--refresh-stale` to
   size a refresh; `evaluate_all(..., dry_run=True)` in code). It runs real factual filtering with
   zero LLM calls and prints exactly how many postings *would* be sent, with a cost estimate from
   recent per-evaluation averages, so cost comes from real counts rather than a guess.
2. State the estimated cost and confirm before running for real, unless the user already asked
   for the full run explicitly.
3. `--limit N` caps real LLM calls (job evals + company classifications) per invocation — use it
   for demos/samples instead of burning through the whole registry.
4. After a real run, report the *actual* cost from `EvaluateStats.estimated_cost_usd()`, not the
   pre-run estimate.
5. A change to the prompts (`jobscan/llm/prompts.py`), the output schemas, the rendered
   candidate profile, or a model or its thinking setting changes the rubric version but does NOT
   re-judge cached verdicts — only `evaluate --refresh-stale` does. (Profile keys the renderer
   doesn't read, such as the code-only `candidate_state` or `excluded_time_zones`, don't change
   it.) Full evaluations and triage screen-outs carry separate versions: a profile change
   re-screens everything, a `JOB_EVAL_SYSTEM`/schema/main-model change only full evaluations, a
   `TRIAGE_SYSTEM`/triage-model change only triage screen-outs. The per-posting message templates
   (`build_*_user_message`) are NOT fingerprinted — a material change there needs a deliberate
   refresh (e.g. a matching edit to the system prompt). Validate a rubric change on a handful of
   known postings first, then dry-run the refresh (`evaluate --dry-run --refresh-stale`) for the
   count.
6. Evaluations keep their history — re-judging adds a row and the newest is current. Reverting a
   rubric change makes the earlier verdicts current again on the next `--refresh-stale` with no API
   calls (a cache hit is copied forward), and before/after comparisons are a query, not a re-run.
7. The cache key is the description hash plus `compute_details_hash` (`jobscan/normalize.py`):
   everything else the prompts print about the posting except its URL. A field newly added to
   `build_*_user_message` must go into it too, or postings that differ only in that field share a
   verdict (observed: Mesh's "United States - Remote" copy carried its New York copy's hybrid
   rejection). Changing its formula invalidates every stored `details_hash`, which would re-judge
   the whole pool; set the column to NULL in the same change so `Database._backfill_details_hash`
   recomputes it from the posting each verdict was judged on, then dry-run for the count.

## LLM output schema — field order is reasoning order

Each LLM call asks for one JSON object (structured outputs, `output_config.format`) with thinking
off, and the model writes it in schema order (verified live on Sonnet 5.5 and Haiku 5.5), so the
property order of `JobEvaluationResult` (`jobscan/llm/schemas.py`) is the order the model reasons
in. Keep the evidence/analysis fields before `verdict`/`worth_applying`: with the verdict first, the
model decided before analysing and then rationalized (it once rejected Pilot and then wrote that it
had no reason to). A test pins this order.

Rules the model must apply while writing a field work far better in that field's `description`
than in the system prompt alone (observed: a rule stated only in the prompt was ignored; repeated
in the field description, it held), and a concrete example beats an abstract rule.

`day_to_day_work`, `work_fit` and `frontend_fit` come first: what the team's engineers actually
do, judged apart from whether the qualifications list can be met. The alternative-language and "technologies like
X" rules (added because Sonnet 5 over-rejected) decide the hiring bar, and Sonnet 5.5 applies them
literally — it called GitHub's npm team a plausible match because the list accepted C#. `work_fit`
caps the report tier in code (`reports.data._report_section`), and `frontend_fit: too_heavy` (more
frontend than the profile accepts) takes a role off the shortlist, as `working_hours_fit` does; the
tiers themselves come from the requirement evidence (`_missing_requirements`), not the verdict
label alone, which Sonnet 5.5 hands out generously.

The evaluation model can't be made deterministic — `temperature` is deprecated for it (the API
returns 400). The same posting under the same rubric has come back Best Bet twice and Growth Bet
once, and a near-miss not-worth-applying twice and worth-applying once. So: judge a rubric change
on several runs of a posting, not one; and near-misses (borderline + not worth applying) get a
second look in `evaluate_all`, keeping the more favourable result.

The schema is converted by `jobscan.llm.client.strict_input_schema` at send time (the
fingerprinted schema is unchanged), so the API guarantees the response's shape; Pydantic still
checks ranges. No tool calls: Sonnet 5.5 rejects a forced `tool_choice`. Which `thinking.type`
turns thinking off depends on the model (`between_tools` on Sonnet 5.5, a 400 everywhere else;
`disabled` on Haiku 5.5), so it's set per model in `config/settings.yaml` beside the model name. A
refusal (`stop_reason: "refusal"`, e.g. on offensive-security postings) raises `LlmCallError`, so
the posting stays awaiting evaluation — never cache it as a reject.

## Triage skips must be substantiated

The cheap triage model can only skip for an allowed category, quoting the posting's own words;
`jobscan.llm.triage.skip_is_substantiated` checks that the quote is really in the posting and sends
anything unsupported to the full evaluation (observed: Haiku skipped good roles on level, salary
arithmetic and invented leadership scope, even when told not to). Beyond that, code only checks the
near-objective categories — Staff+ and manager titles, location wording. Whether a *skill* is truly
required is left to the model: don't add pattern-matching of skill sentences ("Go or Python",
"ideally", "you do not need experience with ...") — it was tried and proved too ambiguous. Every
check there may only turn a skip into a full evaluation — never the reverse.

## Architectural boundaries — keep these separate

- `jobscan/job_family.py` (is this posting even software engineering?) runs at **crawl time**,
  before anything is persisted. Any engineer/developer-type title passes unless it names an
  unambiguously different profession (Sales Engineer, Mechanical Engineer, Developer Advocate,
  ...) — permissive on specialty/seniority (DevOps, QA, Principal, EM titles all pass). Don't go
  back to an allowlist of specific phrases: it silently dropped Product Engineer, Analytics
  Engineer and "Managing Engineer (C#/ASP.NET)" for months.
- Code-side rejections (the job-family gate, `jobscan/parsing.py`, `jobscan/filters.py`) are final
  — no LLM ever sees the posting — so a pattern there must resolve ambiguity toward keeping or
  deferring, never toward rejecting. Observed: "Washington, D.C." in an exclusion list rejected
  every Samsara posting, "$153,000 USD and $214,000 USD" parsed as a flat $153K, double spaces
  read as "no salary". When touching these, measure on the DB's real postings: what newly passes,
  and what newly gets rejected.
- `jobscan/filters.py` (safe factual rejections on postings already in the DB) never judges
  qualification language or job family — only salary/remote/employment-type facts and confirmed
  consulting employers. If a check requires reading and interpreting the posting's prose, it
  belongs in the LLM path (`jobscan/llm/`), not here.
- The crawl fetches boards in worker threads (`crawl_workers`), but only the fetching:
  normalizing, `db.upsert_job` and closing stay on the calling thread, in registry order.
  `evaluate` likewise makes only its LLM calls in worker threads (`evaluate_workers`,
  `jobscan.evaluate._screen`); every read and save stays on the calling thread. The `Database` is
  one SQLite connection — never touch it from a worker thread, in the crawl, an adapter or
  `_screen`.
- Never describe registry coverage as "comprehensive" — always report the exact
  companies/postings actually processed (this is a hard requirement from the original spec, not
  a style preference).

## Typing conventions

- Internal canonical models (`jobscan/models.py`): plain dataclasses, straightforward type hints.
- External data (ATS API responses, LLM tool-call output): Pydantic, validated at the boundary
  (`jobscan/adapters/schemas.py`, `jobscan/llm/schemas.py`). Don't validate internal/trusted data
  with Pydantic just because it's available.
- Avoid `Protocol`/generic-abstraction layers for simple test seams — a plainly-typed `Any`
  injectable parameter with a docstring (see `AnthropicClient.__init__`) is preferred over a
  Protocol hierarchy for something only tests use.
- Allow inference for obvious locals; only annotate a local when the type genuinely isn't
  inferable (e.g. an empty collection literal whose element type matters downstream).

## Dependencies

Prefer a small, well-maintained dependency over hand-rolled logic for anything correctness-
sensitive (e.g. `zoneinfo` + `tzdata` for Seattle time, not manually re-derived DST transition
rules) — a maintained IANA database beats code that goes silently stale if the underlying rules
change. This isn't a blanket "add dependencies freely" license — it's specifically about not
reinventing something with a real, non-obvious correctness surface.

## Registry mutation semantics

`companies import` is additive by default. `--replace` deactivates (reversible, history kept)
companies absent from the imported file; `--delete` permanently removes them and all their job/
evaluation history (irreversible). Only use `--delete` when the user has explicitly said they want
that data gone, not merely that they want to "switch" the registry.

## Output timestamps

Report/audit filenames and their `generated_at` header use Seattle local time (`now_seattle()` /
`filename_timestamp()` in `jobscan/timeutil.py`, `PT` suffix). Database-stored timestamps
(`first_seen_at`, `published_at`, evaluation `created_at`, ...) stay UTC — don't convert those;
only display-facing output at report-generation time is Seattle time.
