# Review instructions

Expectations for reviewers and authors on this repository. The general
development flow, test layering, and lint rules live in
[AGENTS.md](../AGENTS.md), the [full development guide](../.agents/AGENTS.md),
and the [test rules](../tests/AGENTS.md); this file only fixes what applies
during code review.

## Review scope

- Review the actual diff against the base branch, not the commit history of
  the fork. State the severity (blocking / non-blocking) for every comment so
  the author can prioritize.
- Comment on behavior that the diff changes; drive-by findings outside the
  diff belong in a separate issue or PR.

## Verification

- This repository has no CI that runs automatically on pull requests (the
  workflows under `.github/workflows/` target self-hosted NPU runners and do
  not trigger for external forks). Reviewers should therefore ask for
  verification evidence in the target environment instead of waiting for
  check runs: the author states the exact commands used and their results
  (lint, unit tests, and where relevant a real-device run) for the reviewed
  commit.
- Required minimum per PR: `ruff check` and `ruff format --check` clean, and
  the affected test files green in the development container. Model or
  simulator changes additionally run their capture/parity test suites.
- Reviewers may request a re-run on the current head before approval; a
  stale verification from an older commit does not count.

## Communication

- Review rounds proceed in comment order; an unresolved blocking comment
  keeps the PR open. Authors reply on the review thread with what changed
  and reference the fix commit.
- Review findings that require a design change (sharding contracts,
  dependency direction, patch/extension boundaries) are discussed before
  code is pushed; see [architecture doc](../docs/architecture.md) for the
  boundary rules the review enforces.
