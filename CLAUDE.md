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

## Architectural boundaries — keep these separate

- `jobscan/job_family.py` (is this posting even software engineering?) runs at **crawl time**,
  before anything is persisted. It's a coarse title allowlist/denylist — permissive on
  specialty/seniority (DevOps, QA, Principal, EM titles all pass), strict only on unambiguously
  different professions (Sales Engineer, Mechanical Engineer, baristas, ...).
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
