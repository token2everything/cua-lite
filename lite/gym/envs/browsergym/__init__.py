"""BrowserGym CUA-Lite env package (env RUNTIME only).

Importing this package is a no-op beyond marking the package: the env RUNTIME
(``main.py`` — the gymnasium env classes + task registration, needs
``browsergym.core``) is lazy-loaded by the registry via ``...browsergym.main``
(``registry._import_env``), NOT eagerly here.

The model-side bridges for this env (the ``browsergym.generic`` protocol + the
``visualwebarena.goal_image`` agent) live in ``lite/agents/extensions/browsergym/``
and register on the model side (``import lite.agents.extensions``) — they are NOT
imported here, which keeps ``lite.gym`` free of any ``lite.agents`` dependency.
"""
