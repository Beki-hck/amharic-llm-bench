import random

import pytest

from amharic_bench.metrics import chrf, corpus_chrf, token_f1

# Reference values computed with sacreBLEU 2.6.0: CHRF().sentence_score(hyp, [ref]).score
KNOWN = [
    ("ሰላም ነው እንዴት ነህ", "ሰላም ነህ እንዴት ነህ", 55.469),
    ("The cat sat on the mat.", "The cat is on the mat.", 67.1727),
    ("abc", "xyz", 0.0),
    ("a", "abcdefgh", 15.1515),
    ("እኔ ወደ ገበያ ሄድኩ።", "ትናንት ወደ ገበያ ሄጄ ነበር።", 29.55),
]


@pytest.mark.parametrize("hyp, ref, expected", KNOWN)
def test_chrf_matches_sacrebleu_values(hyp, ref, expected):
    assert chrf(hyp, ref) == pytest.approx(expected, abs=1e-3)


def test_corpus_chrf_matches_sacrebleu_value():
    assert corpus_chrf([(h, r) for h, r, _ in KNOWN]) == pytest.approx(44.9132, abs=1e-3)


def test_chrf_against_live_sacrebleu_on_random_text():
    sacrebleu = pytest.importorskip("sacrebleu")
    ref_metric = sacrebleu.metrics.CHRF()
    rng = random.Random(7)
    letters = "ሀለሐመሠረሰሸቀበተቸነኘአከወዘየደጀገጠጨጰጸፀፈፐ "
    for _ in range(200):
        hyp = "".join(rng.choice(letters) for _ in range(rng.randint(1, 40)))
        ref = "".join(rng.choice(letters) for _ in range(rng.randint(1, 40)))
        assert chrf(hyp, ref) == pytest.approx(ref_metric.sentence_score(hyp, [ref]).score, abs=1e-6)


def test_chrf_identity_and_best_reference():
    assert chrf("ቡና ጠጣ", "ቡና ጠጣ") == pytest.approx(100)
    assert chrf("ቡና ጠጣ", ["ውሃ ጠጣ", "ቡና ጠጣ"]) == pytest.approx(100)


def test_token_f1():
    assert token_f1("አዲስ አበባ", "አዲስ አበባ") == 1.0
    assert token_f1("አዲስ", "አዲስ አበባ") == pytest.approx(2 / 3)
    assert token_f1("ጎንደር", "አዲስ አበባ") == 0.0
