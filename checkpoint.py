"""Single, validated checkpoint-loading utility.

Replaces the three raw ``model.load_state_dict(self.ckpt["state_dict"])`` calls in
the trainer/predictor, which assumed the checkpoint shape and gave a cryptic
``KeyError`` when ``"state_dict"`` was absent. Behaviour is otherwise preserved
(``strict=True`` matches PyTorch's default).
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)


def load_model_state(model, ckpt: Mapping[str, Any], *, strict: bool = True):
    """Load weights from a Lightning-style checkpoint dict into ``model``.

    Args:
        model: an ``nn.Module`` (or LightningModule) to load weights into.
        ckpt: a checkpoint mapping that must contain a ``"state_dict"`` entry.
        strict: forwarded to ``load_state_dict`` (default matches PyTorch).

    Returns:
        The ``NamedTuple`` of missing/unexpected keys from ``load_state_dict``.
    """
    if "state_dict" not in ckpt:
        raise KeyError(
            "Checkpoint has no 'state_dict' entry; available keys: "
            f"{sorted(ckpt.keys())}"
        )
    result = model.load_state_dict(ckpt["state_dict"], strict=strict)
    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])
    if missing or unexpected:
        logger.warning(
            "load_model_state: %d missing, %d unexpected keys",
            len(missing),
            len(unexpected),
        )
    return result
