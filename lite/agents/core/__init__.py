"""CUA-Lite agent framework — the four pillars.

The agent-side domain contract + engine, split into pillars: ``action_space``
(the ``@tool`` / [0,1000]-coord action vocabulary + registry), ``adapter``
(trajectory ↔ model-dialect translation), ``agent`` (the run loop + registry),
``protocol`` (multi-turn history shaping). Each pillar's ``base.py`` holds its
abstraction + canonical impl. Concrete model dialects live in
``lite/agents/models/<family>/``; env-specific bridges in
``lite/agents/extensions/<env>/``.
"""
