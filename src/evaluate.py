from __future__ import annotations

from pathlib import Path
import re
import time

import pandas as pd

from data_loader import load_documents
from flat_rag import FlatRetriever
from graph_rag import answer_with_graph, load_graph


QUESTIONS = [
    {
        "question": "Why did US EV sales growth slow in Q1 2024 and how was Tesla involved?",
        "keywords": ["slow", "q1", "tesla", "13.3", "kelley"],
    },
    {
        "question": "Which companies reported strong year-over-year EV sales growth in Q1 2024?",
        "keywords": ["bmw", "cadillac", "ford", "hyundai", "rivian", "vinfast"],
    },
    {
        "question": "How do charging infrastructure concerns affect EV adoption in the United States?",
        "keywords": ["charging", "ports", "2030", "infrastructure", "concerns"],
    },
    {
        "question": "What role do incentives or policy regulations play in EV market growth?",
        "keywords": ["incentives", "regulations", "zev", "ira", "tax"],
    },
    {
        "question": "How are Nikola's hydrogen truck strategy and partners connected?",
        "keywords": ["nikola", "hydrogen", "hyla", "voltera", "trucks"],
    },
    {
        "question": "What evidence links consumer charging satisfaction to future EV sales?",
        "keywords": ["satisfaction", "charging", "consumer", "sales", "adoption"],
    },
    {
        "question": "How did the Inflation Reduction Act influence EV purchases and infrastructure?",
        "keywords": ["inflation", "reduction", "act", "7500", "infrastructure"],
    },
    {
        "question": "What does the corpus say about workplace and public chargers in leading EV markets?",
        "keywords": ["workplace", "public", "chargers", "million", "leading"],
    },
    {
        "question": "How does BloombergNEF describe global EV market direction in 2024?",
        "keywords": ["bloombergnef", "sales", "slowdown", "global", "2024"],
    },
    {
        "question": "What is the relationship between battery demand, China, and EV manufacturing?",
        "keywords": ["battery", "china", "manufacturing", "demand", "cells"],
    },
    {
        "question": "How do EV price cuts relate to demand and Tesla's market position?",
        "keywords": ["prices", "tesla", "demand", "market", "transaction"],
    },
    {
        "question": "Which policies or regulations are connected to zero-emission vehicle adoption?",
        "keywords": ["policy", "regulations", "zev", "zero-emission", "adoption"],
    },
    {
        "question": "How do Ford and General Motors appear in the EV market discussion?",
        "keywords": ["ford", "general motors", "gm", "ev", "production"],
    },
    {
        "question": "What does the dataset say about charging speed and cost preferences?",
        "keywords": ["charging", "speed", "cost", "minutes", "drivers"],
    },
    {
        "question": "How are public chargers and home charging related to EV ownership?",
        "keywords": ["home", "public", "charging", "owners", "residences"],
    },
    {
        "question": "What are the main adoption barriers mentioned for hesitant EV buyers?",
        "keywords": ["hesitant", "range", "battery", "charging", "availability"],
    },
    {
        "question": "How does the corpus connect EV adoption with emissions or oil demand?",
        "keywords": ["emissions", "oil", "demand", "evs", "road"],
    },
    {
        "question": "Which organizations are sources for EV market analysis in the dataset?",
        "keywords": ["cox", "mckinsey", "bloombergnef", "icct", "kelley"],
    },
    {
        "question": "How is California connected to EV regulations and infrastructure in the corpus?",
        "keywords": ["california", "carb", "regulation", "zero-emission", "infrastructure"],
    },
    {
        "question": "How does GraphRAG help answer multi-hop questions compared with Flat RAG on this corpus?",
        "keywords": ["graph", "entities", "relations", "multi-hop", "flat"],
    },
]


def compact(text: str, limit: int = 450) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[: limit - 3] + "..." if len(text) > limit else text


def keyword_score(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lower)


def flat_answer(flat_docs) -> str:
    if not flat_docs:
        return ""
    return " ".join(result.text for result in flat_docs[:2])


def main() -> None:
    output = Path("outputs")
    documents = load_documents("dataset")
    retriever = FlatRetriever.load(output / "flat_retriever.pkl")
    graph = load_graph(output / "knowledge_graph.json")

    rows = []
    for idx, item in enumerate(QUESTIONS, start=1):
        question = item["question"]
        keywords = item["keywords"]

        flat_start = time.perf_counter()
        flat_docs = retriever.search(question, top_k=3)
        flat_latency_ms = round((time.perf_counter() - flat_start) * 1000, 2)
        flat_text = flat_answer(flat_docs)

        graph_start = time.perf_counter()
        graph_answer = answer_with_graph(graph, documents, question, retriever=retriever)
        graph_latency_ms = round((time.perf_counter() - graph_start) * 1000, 2)

        flat_score = keyword_score(flat_text, keywords)
        graph_score = keyword_score(graph_answer.answer, keywords)
        if graph_score > flat_score:
            verdict = "GraphRAG better"
        elif graph_score < flat_score:
            verdict = "Flat RAG better"
        else:
            verdict = "Tie"

        rows.append(
            {
                "id": idx,
                "question": question,
                "expected_keywords": ", ".join(keywords),
                "flat_top_docs": " | ".join(f"{item.doc_id}: {item.title}" for item in flat_docs),
                "flat_answer_excerpt": compact(flat_text),
                "flat_keyword_score": flat_score,
                "flat_latency_ms": flat_latency_ms,
                "graph_seed_entities": ", ".join(graph_answer.seed_entities),
                "graph_answer_excerpt": compact(graph_answer.answer),
                "graph_keyword_score": graph_score,
                "graph_latency_ms": graph_latency_ms,
                "graph_paths": " | ".join(graph_answer.paths[:5]),
                "verdict": verdict,
                "note": "Keyword score is a lightweight proxy; final grading should inspect cited evidence.",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output / "evaluation_results.csv", index=False)
    df.to_csv(output / "benchmark_20_questions.csv", index=False)
    print(df[["id", "question", "flat_keyword_score", "graph_keyword_score", "verdict"]].to_string(index=False))
    print(f"\nSaved: {(output / 'evaluation_results.csv').resolve()}")
    print(f"Saved: {(output / 'benchmark_20_questions.csv').resolve()}")


if __name__ == "__main__":
    main()
