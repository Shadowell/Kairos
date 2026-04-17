"""Kairos — A-share fine-tuning & deployment toolkit for the Kronos foundation model.

Exposes the most common entry points at package root::

    from kairos import KronosTokenizer, Kronos, KronosPredictor
    from kairos import KronosWithExogenous
    from kairos.data import build_features, EXOG_COLS
"""

from .vendor.kronos import (  # noqa: F401
    Kronos,
    KronosPredictor,
    KronosTokenizer,
)
from .models.kronos_ext import KronosWithExogenous, QuantileReturnHead  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "Kronos",
    "KronosTokenizer",
    "KronosPredictor",
    "KronosWithExogenous",
    "QuantileReturnHead",
    "__version__",
]
