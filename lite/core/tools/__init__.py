"""Lite tool-call, schema, and result protocol helpers.

Membership rule: **the facade fronts exactly what someone imports through it.**
A name earns a slot here only once a real ``from lite.core.tools import <name>``
exists; every other name is reached at its owning module
(``lite.core.tools.calls`` / ``.results`` / ``.schemas``), which is where the
rest of the repo already reaches them. Re-exporting more than that is a second
copy of an owner's ``__all__`` that no reader consults.
"""

from lite.core.tools.calls import make_tool_call
from lite.core.tools.schemas import make_tool_schema

__all__ = [
    "make_tool_call",
    "make_tool_schema",
]
