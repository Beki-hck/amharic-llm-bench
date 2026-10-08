from evalforge.scorers import get_scorer
from evalforge.suite import Item

import amharic_bench  # noqa: F401  registers the scorers


def score(scorer, answer, expected, **meta):
    item = Item(id="t", prompt="", scorer=scorer, expected=expected, meta=meta)
    return get_scorer(scorer)(item, answer, None)


def test_qa_accepts_affixes_and_spelling_variants():
    assert score("am_qa", "በአዲስ አበባ", ["አዲስ አበባ"]).passed            # "in Addis Ababa"
    assert score("am_qa", "መልስ፦ ሐዋሳ።", ["ሀዋሳ"]).passed                 # label, punctuation, ሐ/ሀ
    assert score("am_qa", "150 ኪሎ", ["150", "መቶ ሃምሳ"]).passed


def test_qa_rejects_wrong_and_padded_answers():
    assert not score("am_qa", "ጎንደር", ["ደሴ"]).passed
    padded = "ጽሑፉ ስለ ብዙ ከተሞች ይናገራል ደሴ ጎንደር ባሕር ዳር አዲስ አበባ ሐረር ጅማ መቐለ አዳማ ሐዋሳ ድሬዳዋ"
    s = score("am_qa", padded, ["ደሴ"])
    assert not s.passed and "too long" in s.detail


def test_choice_formats():
    assert score("am_choice", "መልስ: B", "B").passed
    assert score("am_choice", "መልሱ፡ C ነው", "C").passed
    assert score("am_choice", "Answer: D", "D").passed
    assert score("am_choice", "ትክክለኛው አማራጭ B) አዲስ አበባ ነው።", "B").passed
    assert not score("am_choice", "አላውቅም", "A").passed


def test_chrf_requires_fidel_for_amharic_targets():
    s = score("chrf", "The market opens early.", ["ገበያው ቅዳሜ ጠዋት በማለዳ ይከፈታል።"], target="am")
    assert s.value == 0 and "not in Amharic script" in s.detail


def test_chrf_strips_chatter():
    ref = "ገበያው ቅዳሜ ጠዋት በማለዳ ይከፈታል።"
    chatty = f"Here is the translation:\n{ref}\n(Note: 'ገበያ' means market.)"
    s = score("chrf", chatty, [ref], target="am")
    assert s.value > 0.85 and s.passed
    assert score("chrf", "Translation: The market opens early on Saturday morning.",
                 ["The market opens early on Saturday morning."], target="en").value == 1.0


def test_chrf_ignores_homophone_spelling_for_amharic():
    assert score("chrf", "ሀኪሙ መጣ", ["ሐኪሙ መጣ"], target="am").value == 1.0
