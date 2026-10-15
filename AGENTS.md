# WebArena-Verified v0.4 generation repository

- Use the self-contained `skills/webarena-verified-v04-generate/SKILL.md` when
  asked to generate data. The repository owns executable implementations.
- Read the owner module before editing; preserve its contracts and vocabulary.
- Do not stage, commit, push, publish, or promote data for training unless asked.
- Fixed initial assets and exactly two declared sites per cross-site task. Only normal authentication before
  execution; task-specific writes belong to the task, never baseline preparation.
- Official 812-task content is available only to the post-generation checker.
  Authoring reuses abstract families and grounds facts on the fixed site.
- Author plans, GT, private evaluators and HAR/trace never enter model input.
- Preserve semantic failures, original handles, provenance and historical attempts.
  Reconcile an unknown submission rather than posting it again.
- Clean up only sandbox/container resources whose run ID and nonce match the
  creation receipt. Do not interfere with other sessions.
- Use `uv run python -B`; do not hand-edit dependencies, caches or virtualenvs.
- Human review and training promotion are separate from machine-verified delivery.

## Current implementation ownership

- The coordinator owns sandbox creation, commands, original handles and release.
- Bounded implementation subagents may edit only their assigned files and run offline checks.
- Runtime implementations must use the declared sites directly; no singular-site compatibility aliases.
- Cross-site tasks must bind visible intermediate entities and source facts to their target use.
- Preserve v0.3 provenance as the imported baseline; record v0.4 release hashes separately.
