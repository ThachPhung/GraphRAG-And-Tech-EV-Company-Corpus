from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Document:
    doc_id: str
    query: str
    title: str
    link: str
    snippet: str
    content: str

    @property
    def searchable_text(self) -> str:
        return "\n".join(
            part
            for part in [self.title, self.snippet, self.content]
            if part and part.strip()
        )


def _field(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    query = _field(r"^Query:\s*(.*?)$", raw)
    title = _field(r"^Title:\s*(.*?)$", raw)
    link = _field(r"^Link:\s*(.*?)$", raw)
    snippet = _field(r"^Snippet:\s*(.*?)$", raw)

    marker = "Full Content:"
    content = raw.split(marker, 1)[1].strip() if marker in raw else raw.strip()
    content = re.sub(r"\n{3,}", "\n\n", content)

    return Document(
        doc_id=path.stem,
        query=query,
        title=title,
        link=link,
        snippet=snippet,
        content=content,
    )


def load_documents(dataset_dir: str | Path) -> list[Document]:
    dataset_path = Path(dataset_dir)
    files = sorted(
        dataset_path.glob("doc_*.txt"),
        key=lambda p: int(re.search(r"\d+", p.stem).group(0)),
    )
    return [parse_document(path) for path in files]


def split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    placeholders = {
        "U.S.": "U<dot>S<dot>",
        "Q1.": "Q1<dot>",
        "Q2.": "Q2<dot>",
        "Q3.": "Q3<dot>",
        "Q4.": "Q4<dot>",
    }
    for source, replacement in placeholders.items():
        compact = compact.replace(source, replacement)
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    for source, replacement in placeholders.items():
        sentences = [sentence.replace(replacement, source) for sentence in sentences]
    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 20
    ]
