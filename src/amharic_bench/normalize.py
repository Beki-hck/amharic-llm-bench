"""Amharic text normalization.

Amharic (Ge'ez/Fidel script) has several letter families that sound the same and
are used interchangeably in everyday writing: ሀ/ሐ/ኀ, ሰ/ሠ, አ/ዐ and ጸ/ፀ. "ሐሙስ" and
"ሀሙስ" are the same word. A scorer that compares raw strings would mark a correct
answer wrong just for the spelling variant, so every comparison in this package
goes through `normalize` first.

Each Fidel letter has 7 base "orders" (vowel forms) laid out consecutively in
Unicode, so a whole family can be folded by shifting code points.
"""

from __future__ import annotations

import re
import unicodedata

# (variant family start, canonical family start, how many forms) -- 8 covers the
# 7 vowel orders plus the labialized (-wa) form where Unicode has one.
_FAMILIES = [
    (0x1210, 0x1200, 7),  # ሐ ሑ ሒ ሓ ሔ ሕ ሖ -> ሀ ሁ ሂ ሃ ሄ ህ ሆ
    (0x1280, 0x1200, 7),  # ኀ ኁ ኂ ኃ ኄ ኅ ኆ -> ሀ ...
    (0x1220, 0x1230, 8),  # ሠ ... ሧ -> ሰ ... ሷ
    (0x12D0, 0x12A0, 7),  # ዐ ዑ ዒ ዓ ዔ ዕ ዖ -> አ ኡ ኢ ኣ ኤ እ ኦ
    (0x1340, 0x1338, 8),  # ፀ ... ፇ -> ጸ ... ጿ
]

_HOMOPHONES = {
    chr(src + i): chr(dst + i) for src, dst, n in _FAMILIES for i in range(n)
}
# The 1st and 4th orders of ሀ and አ are pronounced the same too (ሀገር / ሃገር, አማርኛ / ኣማርኛ).
_HOMOPHONES.update({"ሃ": "ሀ", "ኣ": "አ"})
# Resolve chains such as ኃ -> ሃ -> ሀ so every variant lands on one letter.
for _k, _v in list(_HOMOPHONES.items()):
    while _v in _HOMOPHONES:
        _v = _HOMOPHONES[_v]
    _HOMOPHONES[_k] = _v
_HOMOPHONE_TABLE = str.maketrans(_HOMOPHONES)

_PUNCT = str.maketrans({
    "።": ".", "፣": ",", "፤": ";", "፥": ":", "፦": ":", "፧": "?", "፨": " ",
    "፡": " ",  # traditional word separator
    "“": '"', "”": '"', "«": '"', "»": '"', "‘": "'", "’": "'",
})

# Ethiopic numerals: ፩..፱ = 1..9, ፲..፺ = 10..90, ፻ = 100, ፼ = 10,000.
_UNITS = {chr(0x1369 + i): i + 1 for i in range(9)}
_TENS = {chr(0x1372 + i): (i + 1) * 10 for i in range(9)}
_HUNDRED, _MYRIAD = "፻", "፼"
_ETH_NUM = re.compile("[፩-፼]+")


def ethiopic_to_int(numeral: str) -> int:
    """Convert an Ethiopic numeral to an int, e.g. ፲፱፻፹፯ -> 1987, ፳፻፲፮ -> 2016."""
    total = block = cur = 0  # total: multiples of 10,000; block: < 10,000; cur: < 100
    for ch in numeral:
        if ch in _UNITS:
            cur += _UNITS[ch]
        elif ch in _TENS:
            cur += _TENS[ch]
        elif ch == _HUNDRED:
            block = (block + (cur or 1)) * 100
            cur = 0
        elif ch == _MYRIAD:
            total = (total + block + cur or 1) * 10_000
            block = cur = 0
        else:
            raise ValueError(f"{ch!r} is not an Ethiopic numeral")
    return total + block + cur


_ETHIOPIC_LETTER = re.compile("[ሀ-ፚᎀ-ᎏⶀ-⷟꬀-꬯]")
_LATIN_LETTER = re.compile("[A-Za-z]")


def normalize(text: str, *, homophones: bool = True, punctuation: bool = True,
              numerals: bool = True, lowercase: bool = True) -> str:
    """Return a canonical form of Amharic (or mixed) text for comparison."""
    text = unicodedata.normalize("NFC", text)
    if homophones:
        text = text.translate(_HOMOPHONE_TABLE)
    if punctuation:
        text = text.translate(_PUNCT)
    if numerals:
        text = _ETH_NUM.sub(lambda m: str(ethiopic_to_int(m.group(0))), text)
    if lowercase:
        text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def ethiopic_ratio(text: str) -> float:
    """Share of letters in `text` that are Ethiopic script (digits and punctuation ignored).

    Used to catch a model that answers an Amharic task in English or in
    Latin-letter transliteration ("selam" instead of "ሰላም").
    """
    eth = len(_ETHIOPIC_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    return eth / (eth + latin) if eth + latin else 0.0
