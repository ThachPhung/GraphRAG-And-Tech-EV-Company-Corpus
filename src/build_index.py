from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from data_loader import load_documents
from entity_extraction import OPENAI_USAGE, extract_triples
from flat_rag import FlatRetriever
from graph_rag import build_graph, export_neo4j_csv, save_graph


def build(dataset_dir: str, output_dir: str, use_openai: bool = False) -> None:
    start = time.perf_counter()
    load_dotenv()
    if use_openai and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to .env or export it before running --use-openai."
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    documents = load_documents(dataset_dir)
    total_chars = sum(len(doc.searchable_text) for doc in documents)
    estimated_tokens = round(total_chars / 4)
    triples = []
    for doc in tqdm(documents, desc="Extracting triples"):
        triples.extend(extract_triples(doc, use_openai=use_openai))

    pd.DataFrame([triple.to_dict() for triple in triples]).to_csv(
        output / "triples.csv", index=False
    )

    graph = build_graph(triples)
    save_graph(graph, output)
    export_neo4j_csv(graph, output)

    retriever = FlatRetriever.build(documents)
    retriever.save(output / "flat_retriever.pkl")

    elapsed_seconds = round(time.perf_counter() - start, 3)
    input_cost_per_1m = float(os.getenv("OPENAI_INPUT_COST_PER_1M", "0.15"))
    output_cost_per_1m = float(os.getenv("OPENAI_OUTPUT_COST_PER_1M", "0.60"))
    api_cost = (
        OPENAI_USAGE["prompt_tokens"] / 1_000_000 * input_cost_per_1m
        + OPENAI_USAGE["completion_tokens"] / 1_000_000 * output_cost_per_1m
    )
    metrics = {
        "documents": len(documents),
        "triples": len(triples),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "corpus_characters": total_chars,
        "estimated_corpus_tokens": estimated_tokens,
        "build_time_seconds": elapsed_seconds,
        "llm_extraction_enabled": use_openai,
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini") if use_openai else None,
        "api_requests": OPENAI_USAGE["requests"] if use_openai else 0,
        "api_failed_requests": OPENAI_USAGE["failed_requests"] if use_openai else 0,
        "api_fallback_documents": OPENAI_USAGE["fallback_documents"] if use_openai else 0,
        "api_input_tokens": OPENAI_USAGE["prompt_tokens"] if use_openai else 0,
        "api_output_tokens": OPENAI_USAGE["completion_tokens"] if use_openai else 0,
        "api_total_tokens": OPENAI_USAGE["total_tokens"] if use_openai else 0,
        "input_cost_per_1m_tokens_usd": input_cost_per_1m if use_openai else 0.0,
        "output_cost_per_1m_tokens_usd": output_cost_per_1m if use_openai else 0.0,
        "estimated_api_cost_usd": round(api_cost, 6) if use_openai else 0.0,
        "note": (
            "OpenAI token usage is collected from API responses."
            if use_openai
            else "Offline rule-based extraction was used, so API token cost is zero."
        ),
    }
    (output / "build_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Documents: {len(documents)}")
    print(f"Triples: {len(triples)}")
    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    print(f"Estimated corpus tokens: {estimated_tokens}")
    print(f"Build time: {elapsed_seconds}s")
    if use_openai:
        print(f"OpenAI requests: {OPENAI_USAGE['requests']}")
        print(f"OpenAI total tokens: {OPENAI_USAGE['total_tokens']}")
        print(f"Estimated API cost: ${api_cost:.6f}")
    print(f"Saved outputs to: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Flat RAG and GraphRAG indexes.")
    parser.add_argument("--dataset", default="dataset", help="Folder containing doc_*.txt files")
    parser.add_argument("--output", default="outputs", help="Output folder for indexes")
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI for triple extraction. Falls back to rules on failure.",
    )
    args = parser.parse_args()
    build(args.dataset, args.output, use_openai=args.use_openai)


if __name__ == "__main__":
    main()
