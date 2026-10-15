"""WebGym CUA-Lite env package (env RUNTIME only).

Importing this package is a no-op beyond marking the package: the env RUNTIME
(``main`` — needs httpx / the OmniBoxes client) is lazy-loaded by the registry
via ``...webgym.main`` (``registry._import_env``), NOT eagerly here.

WebGym rows run on the agent family's own history protocol; there is no
env-specific model-side bridge, which keeps ``lite.gym`` free of any
``lite.agents`` dependency. Recommended rows live in
``scripts/configs/<family>/default/webgym.yaml``.
"""
