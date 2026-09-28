"""
Shared text helpers for the Arabic / honorific fine-tuning tools
(make_lines.py, page_to_lines.py, eval_lines.py, check_font.py). No torch dependency.

Label conventions used throughout:
  * labels are NFC-normalised, so shadda/vowel order and composed letters (آ أ ؤ إ ۂ) are
    always stored the same way - CTC cannot learn two spellings of one image
  * invisible formatting characters (ZWNJ, ZWJ, direction marks, BOM) and tatweel are removed
  * honorifics are labelled as the full phrase (صلی اللہ علیہ وسلم), however they are printed
"""
import re
import unicodedata

HARAKAT = set("ًٌٍَُِّْٰٕٖٓٔٗ٘")

# zero-width / bidi controls and tatweel: never visible as separate glyphs, so never in labels
_INVISIBLE = re.compile("[​-‏‪-‮⁦-⁩﻿ـ]")

# letters that only occur in one of the two languages (used to guess a line's language)
URDU_ONLY = set("ٹڈڑںھہےۓگچپژکی")
ARABIC_ONLY = set("ةيكهىإٱ")

# label phrase -> glyphs a font may use to print it compactly.
#   "lig":  a standalone ligature codepoint (ﷺ, or the Unicode 14 honorific ligatures)
#   "mark": a combining sign drawn over the end of the preceding name (ابوہریرہؓ)
HONORIFICS = {
    "صلی اللہ علیہ وسلم":        {"lig": ["ﷺ"], "mark": "ؐ"},
    "صلی اللہ علیہ وآلہ وسلم":   {"lig": ["﵆", "﵌"], "mark": None},
    "رضی اللہ عنہ":              {"lig": ["﵁"], "mark": "ؓ"},
    "رضی اللہ عنہا":             {"lig": ["﵂"], "mark": None},
    "رضی اللہ عنہم":             {"lig": ["﵃"], "mark": None},
    "رضی اللہ عنہما":            {"lig": ["﵄"], "mark": None},
    "رحمۃ اللہ علیہ":            {"lig": [], "mark": "ؒ"},
    "رحمہ اللہ":                 {"lig": ["﵀"], "mark": None},
    "علیہ السلام":               {"lig": ["﵇"], "mark": "ؑ"},
    "علیہا السلام":              {"lig": ["﵍"], "mark": None},
    "علیہم السلام":              {"lig": ["﵈"], "mark": None},
    "جل جلالہ":                  {"lig": ["ﷻ"], "mark": None},
}

# printed ligature / mark -> the phrase used in labels (for turning typed text into labels)
EXPAND = {"ﷺ": "صلی اللہ علیہ وسلم", "ؐ": "صلی اللہ علیہ وسلم",
          "﵁": "رضی اللہ عنہ", "ؓ": "رضی اللہ عنہ",
          "﵂": "رضی اللہ عنہا", "﵃": "رضی اللہ عنہم", "﵄": "رضی اللہ عنہما",
          "ؒ": "رحمۃ اللہ علیہ", "﵀": "رحمہ اللہ",
          "﵇": "علیہ السلام", "ؑ": "علیہ السلام", "﵍": "علیہا السلام",
          "﵈": "علیہم السلام", "﵆": "صلی اللہ علیہ وآلہ وسلم",
          "﵌": "صلی اللہ علیہ وآلہ وسلم", "ﷻ": "جل جلالہ"}


def clean(text, expand_honorifics=True):
    """Normalise a label: honorific signs -> phrases, drop invisibles, NFC, single spaces."""
    text = _INVISIBLE.sub("", text)
    if expand_honorifics:
        text = "".join(f" {EXPAND[c]} " if c in EXPAND else c for c in text)
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def load_charset(path):
    """Same parsing as train.py/read.py: one glyph per line, plus a trailing space character."""
    with open(path, encoding="utf-8") as f:
        return "".join(line.strip("\n") for line in f) + " "


def outside(text, charset):
    """Characters of `text` the model cannot output (sorted, unique)."""
    allowed = set(charset)
    return sorted({c for c in text if c not in allowed})


def strip_harakat(text):
    return "".join(c for c in text if c not in HARAKAT)


def is_arabic(text):
    """Heuristic: vowelled text or Arabic-only letters -> Arabic matn; otherwise Urdu."""
    letters = [c for c in text if "؀" <= c <= "ۿ" and c not in HARAKAT]
    if not letters:
        return False
    harakat = sum(c in HARAKAT for c in text)
    if harakat / len(letters) > 0.25:
        return True
    return sum(c in ARABIC_ONLY for c in letters) > sum(c in URDU_ONLY for c in letters)


def honorifics_in(text):
    """Honorific phrases occurring in a label, longest first, each position counted once."""
    found, taken = [], [False] * len(text)
    for phrase in sorted(HONORIFICS, key=len, reverse=True):
        start = text.find(phrase)
        while start != -1:
            if not any(taken[start:start + len(phrase)]):
                found.append(phrase)
                for i in range(start, start + len(phrase)):
                    taken[i] = True
            start = text.find(phrase, start + 1)
    return found


def edit_distance(a, b):
    """Levenshtein distance between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
