from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import Document


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    title: str
    link: str
    score: float
    text: str


class FlatRetriever:
    def __init__(self, vectorizer: TfidfVectorizer, matrix, documents: list[Document]):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.documents = documents

    @classmethod
    def build(cls, documents: list[Document]) -> "FlatRetriever":
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=20000,
        )
        matrix = vectorizer.fit_transform([doc.searchable_text for doc in documents])
        return cls(vectorizer, matrix, documents)

    def search(self, question: str, top_k: int = 5) -> list[SearchResult]:
        query_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [
            SearchResult(
                doc_id=self.documents[i].doc_id,
                title=self.documents[i].title,
                link=self.documents[i].link,
                score=float(scores[i]),
                text=self.documents[i].searchable_text[:1200],
            )
            for i in ranked
        ]

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "FlatRetriever":
        with Path(path).open("rb") as f:
            return pickle.load(f)
