"""amharic-llm-bench: an Amharic benchmark for LLMs, built on evalforge."""

from . import scorers  # noqa: F401  registers am_qa, am_choice and chrf with evalforge
from .metrics import chrf, corpus_chrf, token_f1
from .normalize import ethiopic_ratio, ethiopic_to_int, normalize

__all__ = ["chrf", "corpus_chrf", "token_f1", "normalize", "ethiopic_ratio", "ethiopic_to_int"]
__version__ = "0.1.0"
