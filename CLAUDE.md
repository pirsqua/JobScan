# Operating notes for Claude Code in this repo

For what the project *is* and how to run it, read `README.md` — this file is only for things a
fresh session needs before touching code, that reading the code alone won't make obvious.

## Environment

- Windows, PowerShell primary; venv at `.venv` (`.\.venv\Scripts\Activate.ps1`).
- Python 3.14. Run the suite before considering any change done: `python -m pytest -q`.

## Cost awareness — `evaluate` and `run` spend real money

They call the real Anthropic API (model/pricing in `config/settings.yaml`). Before running either
without `--limit` against the full registry:
1. Do a free dry run first — build `Settings` with `anthropic_api_key=None` (via
   `dataclasses.replace(settings, anthropic_api_key=None)`) and call `evaluate_all()` — this runs
   real factual filtering with zero LLM calls and reports exactly how many postings *would* be
   sent, so cost can be estimated from real counts rather than a guess.
2. State the estimated cost and confirm before running for real, unless the user already asked
   for the full run explicitly.
3. `--limit N` caps real LLM calls (job evals + company classifications) per invocation — use it
   for demos/samples instead of burning through the whole registry.
4. After a real run, report the *actual* cost from `EvaluateStats.estimated_cost_usd()`, not the
   pre-run estimate.
5. A change to the prompts (`jobscan/llm/prompts.py`), the tool schemas or
   `config/candidate_profile.yaml` changes the rubric version but does NOT re-judge cached
   verdicts — only `evaluate --refresh-stale` does. Full evaluations and triage screen-outs carry
   separate versions: a profile change re-screens everything, a `JOB_EVAL_SYSTEM`/schema change
   only full evaluations, a `TRIAGE_SYSTEM` change only triage screen-outs. The per-posting
   message templates (`build_*_user_message`) are NOT fingerprinted — a material change there
   needs a deliberate refresh (e.g. a matching edit to the system prompt). Validate a rubric change on a handful of known postings first, then dry-run the refresh
   (`evaluate_all(..., refresh_stale=True)` with no key) for the count.

## LLM output schema — field order is reasoning order

The evaluation tool call is forced with no separate thinking step, so the property order of
`JobEvaluationResult` (`jobscan/llm/schemas.py`) is the order the model reasons in. Keep the
evidence/analysis fields before `verdict`/`worth_applying`: with the verdict first, the model
decided before analysing and then rationalized (it once rejected Pilot and then wrote that it had
no reason to). A test pins this order.

Rules the model must apply while writing a field work far better in that field's `description`
than in the system prompt alone (observed: a rule stated only in the prompt was ignored; repeated
in the field description, it held), and a concrete example beats an abstract rule.

Tool calls go out in strict mode (`strict: true`, schema converted by
`jobscan.llm.client.strict_input_schema` at send time — the fingerprinted schema is unchanged), so
the API guarantees the response's shape; Pydantic still checks ranges.

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
