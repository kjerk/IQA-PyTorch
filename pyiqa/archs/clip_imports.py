"""Shim that provides a unified ``clip`` namespace backed entirely by vendored code.

Previously this imported the external ``openai-clip`` package (which pulls in
``pkg_resources`` and breaks on Python 3.13+).  All consumers of this module
use one or more of:

* ``clip.tokenize(...)``       — text tokenization
* ``clip.load(name, ...)``     — CLIP model loading (returns ``(model, None)``)
* ``SimpleTokenizer``          — low-level BPE tokenizer

All three are now served from the vendored modules ``clip_tokenizer`` and
``clip_model`` so no external ``clip`` package is required.
"""

from types import SimpleNamespace

from .clip_tokenizer import SimpleTokenizer, tokenize
from .clip_model import load as _load_model


def _load(name, device='cpu', jit=False, download_root=None):
    """Thin wrapper around the vendored ``clip_model.load`` that returns a
    ``(model, None)`` tuple matching the openai-clip API."""
    model = _load_model(name, device=device, jit=jit, download_root=download_root)
    return model, None


clip = SimpleNamespace(tokenize=tokenize, load=_load, SimpleTokenizer=SimpleTokenizer)

__all__ = ['clip', 'SimpleTokenizer']