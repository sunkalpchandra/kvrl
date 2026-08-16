"""Trace collection and storage (see .claude/context/DATA_SPEC.md)."""

from .storage import Trace, load_trace, save_trace, trace_index

__all__ = ["Trace", "load_trace", "save_trace", "trace_index"]
