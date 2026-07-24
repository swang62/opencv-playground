import re

# Common question wrappers and command prefixes to strip from input.
WRAPPER_PATTERNS = [
    r"where\s+is\b",
    r"where\s+are\b",
    r"where'?s\b",
    r"what\s+is\b",
    r"what\s+are\b",
    r"what'?s\b",
    r"could\s+you\s+find\b",
    r"could\s+you\s+spot\b",
    r"can\s+you\s+find\b",
    r"can\s+you\s+see\b",
    r"do\s+you\s+see\b",
    r"i'?m\s+looking\s+for\b",
    r"i\s+am\s+looking\s+for\b",
    r"show\s+me\b",
    r"look\s+for\b",
    r"search\s+for\b",
    r"find\b",
    r"i\s+want\b",
    r"i\s+need\b",
]

FILLER_WORDS_RE = re.compile(r"\b(a|an|the|my|your|some)\b", re.IGNORECASE)


def normalize_query(raw: str) -> str | None:
    """Reduce a natural-language query to a concise target phrase.

    Strips common question wrappers (e.g. "where is", "find"),
    filler words ("a", "an", "the", "my", "your", "some"), and
    punctuation while preserving meaningful descriptors.
    Returns None when the input reduces to nothing.
    """
    text = raw.strip()
    if not text:
        return None

    for pattern in WRAPPER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = FILLER_WORDS_RE.sub("", text)
    text = re.sub(r"[^\w\s-]", "", text)  # remove punctuation, keep hyphens
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else None
