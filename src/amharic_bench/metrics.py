"""Text-similarity metrics for Amharic output.

chrF (Popović, 2015) compares character n-grams instead of words. That suits
Amharic, where one word carries prefixes and suffixes ("በቤታችን" = "in our house")
and word-level metrics like BLEU punish a translation for a different affix. This
is a re-implementation of sacreBLEU's default chrF (6 character orders, beta = 2,
whitespace ignored), checked against sacreBLEU in the tests.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

CHAR_ORDER = 6
BETA = 2.0


def _ngrams(text: str, n: int) -> Counter:
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


@dataclass
class ChrfStats:
    """Per-order counts: (hyp n-grams, ref n-grams, matches) for n = 1..6."""

    counts: list[tuple[int, int, int]]

    def __add__(self, other: "ChrfStats") -> "ChrfStats":
        return ChrfStats([tuple(a + b for a, b in zip(x, y)) for x, y in zip(self.counts, other.counts)])

    def score(self) -> float:
        """chrF on a 0-100 scale, computed the way sacreBLEU does it."""
        eps = 1e-16
        beta2 = BETA ** 2
        avg_p = avg_r = 0.0
        effective = 0
        for hyp, ref, match in self.counts:
            if hyp > 0 and ref > 0:
                avg_p += match / hyp
                avg_r += match / ref
                effective += 1
        if effective == 0:
            return 0.0
        avg_p /= effective
        avg_r /= effective
        if avg_p + avg_r == 0:
            return 0.0
        return 100 * (1 + beta2) * avg_p * avg_r / (beta2 * avg_p + avg_r + eps)


def chrf_stats(hypothesis: str, reference: str) -> ChrfStats:
    hyp = re.sub(r"\s+", "", hypothesis)
    ref = re.sub(r"\s+", "", reference)
    counts = []
    for n in range(1, CHAR_ORDER + 1):
        h, r = _ngrams(hyp, n), _ngrams(ref, n)
        counts.append((sum(h.values()), sum(r.values()), sum((h & r).values())))
    return ChrfStats(counts)


def chrf(hypothesis: str, references: str | list[str]) -> float:
    """Sentence chrF against one or more references (the best-matching reference wins)."""
    refs = [references] if isinstance(references, str) else references
    return max(chrf_stats(hypothesis, r).score() for r in refs)


def corpus_chrf(pairs: list[tuple[str, str]]) -> float:
    """Corpus chrF: sum the n-gram counts over all sentences, then score once."""
    if not pairs:
        return 0.0
    total = chrf_stats(*pairs[0])
    for hyp, ref in pairs[1:]:
        total = total + chrf_stats(hyp, ref)
    return total.score()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text)


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style word-overlap F1 (inputs should already be normalized)."""
    pred, ref = _tokens(prediction), _tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pred), common / len(ref)
    return 2 * p * r / (p + r)
