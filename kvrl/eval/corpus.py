"""Natural-text filler corpus for long-context tasks.

We use public-domain Project Gutenberg books (downloaded on demand into
``data/corpus/``, gitignored) as haystack text so attention patterns during
evaluation resemble real prose rather than templated filler. If the download is
impossible (offline), a deterministic template generator is used and the task
metadata records ``filler="template"`` so results are never silently conflated.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

DEFAULT_BOOKS = {
    # id: title (all public domain)
    11: "Alice's Adventures in Wonderland",
    1342: "Pride and Prejudice",
    2701: "Moby Dick",
    84: "Frankenstein",
}
CORPUS_DIR = Path("data/corpus")

_START_RE = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
_END_RE = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)


def _strip_gutenberg(text: str) -> str:
    m = _START_RE.search(text)
    if m:
        text = text[m.end() :]
    m = _END_RE.search(text)
    if m:
        text = text[: m.start()]
    text = text.replace("\r\n", "\n")
    # collapse hard-wrapped lines into paragraphs
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if len(p) > 40 and not p.isupper()]
    return "\n\n".join(paras)


def download_book(book_id: int, corpus_dir: Path = CORPUS_DIR, timeout: float = 30) -> Path:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    p = corpus_dir / f"pg{book_id}.txt"
    if p.exists() and p.stat().st_size > 10_000:
        return p
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        p.write_bytes(r.read())
    return p


def load_corpus(
    corpus_dir: Path = CORPUS_DIR, book_ids: list[int] | None = None, allow_download: bool = True
) -> dict[int, str]:
    """Return {book_id: cleaned text}. Missing books are downloaded if allowed."""
    out: dict[int, str] = {}
    for bid in book_ids or list(DEFAULT_BOOKS):
        p = corpus_dir / f"pg{bid}.txt"
        if not p.exists() and allow_download:
            try:
                download_book(bid, corpus_dir)
            except Exception:
                continue
        if p.exists():
            out[bid] = _strip_gutenberg(p.read_text(errors="ignore"))
    return out


_TEMPLATE_SUBJECTS = [
    "The committee",
    "A quiet river",
    "The old engineer",
    "Our neighbour",
    "The northern road",
    "A small bakery",
    "The archive",
    "The harbour master",
]
_TEMPLATE_VERBS = [
    "reviewed",
    "described",
    "measured",
    "remembered",
    "repainted",
    "catalogued",
    "avoided",
    "celebrated",
]
_TEMPLATE_OBJECTS = [
    "the annual report",
    "a row of lanterns",
    "the tide tables",
    "an unusual map",
    "the winter schedule",
    "several brass keys",
    "the eastern wall",
    "a forgotten ledger",
]
_TEMPLATE_TAILS = [
    "before the rain returned.",
    "without much ceremony.",
    "late in the afternoon.",
    "as the bells rang twice.",
    "and nobody objected.",
    "for the third time that year.",
]


def template_filler(n_chars: int, seed: int = 0) -> str:
    """Deterministic pseudo-natural filler used only when no corpus is available."""
    import random

    rng = random.Random(seed)
    parts: list[str] = []
    total = 0
    while total < n_chars:
        s = (
            f"{rng.choice(_TEMPLATE_SUBJECTS)} {rng.choice(_TEMPLATE_VERBS)} "
            f"{rng.choice(_TEMPLATE_OBJECTS)} {rng.choice(_TEMPLATE_TAILS)}"
        )
        parts.append(s)
        total += len(s) + 1
    return " ".join(parts)


class Filler:
    """Sample contiguous natural-text passages of a requested character length."""

    def __init__(self, corpus: dict[int, str] | None = None, seed: int = 0):
        import random

        self.rng = random.Random(seed)
        self.paragraphs: list[str] = []
        self.source = "template"
        if corpus:
            for text in corpus.values():
                self.paragraphs.extend(p for p in text.split("\n\n") if 40 < len(p) < 2000)
            if self.paragraphs:
                self.source = "gutenberg"

    def sample(self, n_chars: int) -> str:
        if self.source == "template":
            return template_filler(n_chars, seed=self.rng.randrange(1 << 30))
        start = self.rng.randrange(len(self.paragraphs))
        out: list[str] = []
        total = 0
        i = start
        while total < n_chars:
            p = self.paragraphs[i % len(self.paragraphs)]
            out.append(p)
            total += len(p) + 2
            i += 1
        text = "\n\n".join(out)
        return text[:n_chars].rsplit(" ", 1)[0] if len(text) > n_chars else text
