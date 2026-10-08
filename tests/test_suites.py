"""The data itself is tested: every gold answer must get full marks from its own scorer."""

from collections import Counter

import pytest
from evalforge.scorers import get_scorer

from amharic_bench.normalize import ethiopic_ratio
from amharic_bench.suites import SUITES, load


@pytest.mark.parametrize("name", list(SUITES))
def test_suite_loads_with_system_prompt(name):
    suite = load(name)
    assert suite.system and len(suite.items) >= 10
    assert len({it.prompt for it in suite.items}) == len(suite.items), "duplicate prompts"


@pytest.mark.parametrize("name", ["am_qa", "en_am", "am_en", "am_sum"])
def test_gold_answers_score_full_marks(name):
    for it in load(name).items:
        for gold in it.expected:
            s = get_scorer(it.scorer)(it, gold, None)
            assert s.passed and s.value == pytest.approx(1.0), (it.id, gold, s.detail)


def test_mcq_gold_letters_pass_and_are_spread_out():
    items = load("am_mcq").items
    for it in items:
        assert get_scorer(it.scorer)(it, f"መልስ: {it.expected}", None).passed
        options = [ln[3:] for ln in it.prompt.splitlines() if ln[:3] in ("A) ", "B) ", "C) ", "D) ")]
        assert len(options) == 4 and len(set(options)) == 4, it.id
    counts = Counter(it.expected for it in items)
    assert set(counts) == set("ABCD") and max(counts.values()) <= len(items) / 2


def test_amharic_text_is_in_fidel():
    for name in ("am_qa", "am_mcq", "am_sum"):
        for it in load(name).items:
            assert ethiopic_ratio(it.prompt.splitlines()[0]) > 0.8, it.id  # options may be English
    for it in load("en_am").items + load("am_sum").items:
        assert all(ethiopic_ratio(ref) > 0.95 for ref in it.expected), it.id


def test_translation_directions_use_the_same_pairs():
    en_am, am_en = load("en_am").items, load("am_en").items
    assert len(en_am) == len(am_en)
    for a, b in zip(en_am, am_en):
        assert a.expected[0] in b.prompt and b.expected[0] in a.prompt
