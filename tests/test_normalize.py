import pytest

from amharic_bench.normalize import ethiopic_ratio, ethiopic_to_int, normalize


@pytest.mark.parametrize("a, b", [
    ("ሐሙስ", "ሀሙስ"),          # ሐ / ሀ
    ("ኃይል", "ሀይል"),          # ኃ -> ሃ -> ሀ (chained)
    ("ሠላም", "ሰላም"),          # ሠ / ሰ
    ("ዓመት", "አመት"),          # ዓ (4th order of ዐ) / አ
    ("ፀሐይ", "ጸሀይ"),          # ፀ / ጸ
    ("ሃገር", "ሀገር"),          # 1st / 4th order of ሀ
])
def test_homophone_spellings_match(a, b):
    assert normalize(a) == normalize(b)


def test_distinct_letters_stay_distinct():
    assert normalize("ሰላም") != normalize("ሸላም")
    assert normalize("ቤት") != normalize("ቢት")


def test_punctuation_and_whitespace():
    assert normalize("ሰላም፣ እንዴት ነህ፧  ደህና ነኝ።") == "ሰላም, እንዴት ነህ? ደህና ነኝ."
    assert normalize("ሰላም፡ነው") == "ሰላም ነው"


@pytest.mark.parametrize("numeral, value", [
    ("፩", 1), ("፲", 10), ("፲፪", 12), ("፻", 100), ("፪፻", 200),
    ("፲፱፻፹፯", 1987), ("፳፻፲፮", 2016), ("፼", 10_000), ("፫፼፭፻", 30_500),
])
def test_ethiopic_numerals(numeral, value):
    assert ethiopic_to_int(numeral) == value


def test_numerals_inside_text():
    assert normalize("በ፲፱፻፹፯ ዓ.ም") == "በ1987 አ.ም"


def test_ethiopic_ratio():
    assert ethiopic_ratio("ሰላም ነው") == 1.0
    assert ethiopic_ratio("selam new") == 0.0
    assert 0.4 < ethiopic_ratio("ሰላም hi") < 0.7
    assert ethiopic_ratio("123 ።") == 0.0
