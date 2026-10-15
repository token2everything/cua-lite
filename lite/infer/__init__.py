"""cua-lite infer — inference-time entrypoint (eval / collection / weight serving).

ENTRYPOINT tier: orchestrates the subsystems (``gym`` / ``agents``) to run
agents on tasks at scale. Imported by nothing inside ``lite/`` (only by
``scripts/``); may import any subsystem/foundation but never the ``train``
entrypoint.

Modules:
    - rollout: ``run_rollout`` (the run loop) + task collection + report + CI gates
    - cli:     ``run_infer`` + ``make_infer_parser`` — the unified entry the single
      ``scripts/rollout.py`` shell calls (local/API × single-task/all-tasks)
    - serving: sglang generate_fn wrappers + ``local_model`` (model-weight serving)
"""
