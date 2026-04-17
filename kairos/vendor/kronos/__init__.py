"""Vendored copy of the Kronos reference implementation.

Source: https://github.com/shiyu-coder/Kronos
License: MIT (see upstream LICENSE)

We vendor the two model source files so that ``kairos`` is a standalone,
pip-installable project without needing users to clone Kronos separately.
Public classes are re-exported at :mod:`kairos.vendor.kronos`.
"""

from .kronos import (  # noqa: F401
    Kronos,
    KronosPredictor,
    KronosTokenizer,
    auto_regressive_inference,
    calc_time_stamps,
    sample_from_logits,
    top_k_top_p_filtering,
)
from . import module  # noqa: F401

__all__ = [
    "Kronos",
    "KronosTokenizer",
    "KronosPredictor",
    "auto_regressive_inference",
    "calc_time_stamps",
    "sample_from_logits",
    "top_k_top_p_filtering",
    "module",
]
