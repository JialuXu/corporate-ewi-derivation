"""Prompt templates for analysis-layer LLM calls.

We keep prompts as separate string constants (mirroring l4_explain/prompt.py)
so they can be diffed cleanly and translated/extended without editing the
business logic in meta_report.py / suspicion.py.
"""
from .meta_report import META_REPORT_SYSTEM_PROMPT, assemble_meta_report_prompt
from .suspicion import SUSPICION_SYSTEM_PROMPT, assemble_suspicion_prompt

__all__ = [
    "META_REPORT_SYSTEM_PROMPT",
    "SUSPICION_SYSTEM_PROMPT",
    "assemble_meta_report_prompt",
    "assemble_suspicion_prompt",
]
