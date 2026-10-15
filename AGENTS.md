# WebArena-Verified v0.3 generation repository

- Use the self-contained `skills/webarena-verified-v03-generate/SKILL.md` when
  asked to generate data. The repository owns executable implementations.
- Read the owner module before editing; preserve its contracts and vocabulary.
- Do not stage, commit, push, publish, or promote data for training unless asked.
- Fixed initial assets and one site per task. Only normal authentication before
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
