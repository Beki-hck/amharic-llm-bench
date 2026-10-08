"""Amharic-aware scorers, registered into evalforge's scorer registry.

Importing this module is enough: evalforge's `@scorer` decorator adds them, so
suites can say `"scorer": "am_qa"` and evalforge's runner, cache and reports
work unchanged.
"""

from __future__ import annotations

import re

from evalforge.scorers import Score, _expected_list, scorer

from .metrics import chrf, token_f1
from .normalize import ethiopic_ratio, normalize

# An answer counts as "in Amharic" when at least this share of its letters is Fidel.
MIN_ETHIOPIC = 0.5


def _strip_preamble(answer: str) -> str:
    """Drop a leading 'Answer:' / 'መልስ:' label and surrounding quotes or markdown."""
    text = answer.strip().strip("*`").strip()
    text = re.sub(r"^(?:answer|translation|summary|መልስ|ትርጉም|ማጠቃለያ)\s*[:፡፦-]\s*", "", text, flags=re.I)
    return text.strip().strip('"“”«»*').strip()


def _clean_output(answer: str, target: str | None) -> str:
    """Remove chatter around a translation or summary.

    Drops a lead-in line such as "Here is the translation:". For Amharic targets
    it keeps only the lines that are mostly Fidel, so an English note after the
    translation doesn't count against it (an all-English reply still fails).
    """
    lines = [ln for ln in answer.strip().splitlines() if ln.strip()]
    if len(lines) > 1 and lines[0].rstrip().endswith((":", "፡", "፦")):
        lines = lines[1:]
    if target == "am":
        fidel = [ln for ln in lines if ethiopic_ratio(ln) >= MIN_ETHIOPIC]
        lines = fidel or lines
    return _strip_preamble("\n".join(lines))


@scorer("am_qa")
def am_qa(item, answer, ctx) -> Score:
    """Short-answer QA. Passes if the normalized gold answer appears in a short reply,
    or if word-overlap F1 with a gold answer is at least 0.5.

    The containment rule handles Amharic affixes: "በአዲስ አበባ" ("in Addis Ababa")
    contains the gold "አዲስ አበባ". It only applies to short replies, so a model
    can't pass by pasting the whole passage back.
    """
    got = normalize(_strip_preamble(answer))
    golds = [normalize(e) for e in _expected_list(item)]
    max_words = int(item.meta.get("max_words", 12))
    short = len(got.split()) <= max(max_words, 3 * max(len(g.split()) for g in golds))
    contained = short and any(g in got for g in golds)
    f1 = max(token_f1(got, g) for g in golds)
    value = 1.0 if contained else f1
    passed = contained or f1 >= 0.5
    why = "contains gold" if contained else f"F1 {f1:.2f}" + ("" if short else " (reply too long)")
    return Score(passed, value, f"{why}: {got[:50]!r}")


_AM_CHOICE = re.compile(r"(?:answer|መልስ(?:ው)?)\s*(?:is|:|፡|፦)?\s*\(?([A-D])\b", re.I)


@scorer("am_choice")
def am_choice(item, answer, ctx) -> Score:
    """Multiple choice with options A-D. Accepts 'Answer: B', 'መልስ፡ B' or a lone letter."""
    m = _AM_CHOICE.search(answer)
    letters = [m.group(1)] if m else re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", answer)
    if not letters:
        return Score(False, 0.0, "no option letter found")
    got = letters[-1].upper()
    ok = got in {e.upper() for e in _expected_list(item)}
    return Score(ok, float(ok), f"picked {got}")


@scorer("chrf")
def chrf_scorer(item, answer, ctx) -> Score:
    """Translation / summarization: chrF against reference(s), value on a 0-1 scale.

    If the item says `"target": "am"`, the reply must be mostly Fidel script; an
    English or transliterated reply scores 0 instead of getting partial credit
    for shared names and numbers. `pass_chrf` (default 40) only sets the
    pass/fail mark; the leaderboard reports the chrF itself.
    """
    text = _clean_output(answer, item.meta.get("target"))
    if item.meta.get("target") == "am":
        ratio = ethiopic_ratio(text)
        if ratio < MIN_ETHIOPIC:
            return Score(False, 0.0, f"not in Amharic script ({ratio:.0%} Fidel)")
        hyp, refs = normalize(text), [normalize(r) for r in _expected_list(item)]
    else:
        hyp, refs = normalize(text, homophones=False), [normalize(r, homophones=False) for r in _expected_list(item)]
    score = chrf(hyp, refs)
    threshold = float(item.meta.get("pass_chrf", 40))
    return Score(score >= threshold, score / 100, f"chrF {score:.1f}")
