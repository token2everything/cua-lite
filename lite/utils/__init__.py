"""Low-semantic utility namespace.

Import concrete helpers from their owning modules, e.g.
``lite.utils.registry`` or ``lite.utils.image``.

Only pure path, formatting, collection, and serialization helpers with no
runtime policy belong here. Every module is classified:

  * pure utilities: ``config``, ``git``, ``logging``, ``parquet``, ``path``,
    ``timer``;
  * documented cross-layer contracts: ``image``, which owns the canonical
    resize geometry the model families project from, and ``registry``, which
    owns the registry key grammar. Both have many owners across ``lite`` and no
    narrower home.

Semantic runtime policy lives with its owner instead, never here: agent
provider-call retry in ``lite.agents.core.agent.utils.retry``, env-server HTTP
retry in ``lite.gym.remote``, and the CUAWorld VLM judge's backoff in that
env's ``src/vlm.py``.

This package initializer stays import-light and exposes no re-export facade.
"""
