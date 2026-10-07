"""Explicit single-card LoRA backend; not the upstream SLIME training backend."""

from .backend import load_agent, update_from_group

__all__ = ["load_agent", "update_from_group"]
