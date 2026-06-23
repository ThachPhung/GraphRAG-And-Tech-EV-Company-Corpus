from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from data_loader import load_documents
from flat_rag import FlatRetriever
from graph_rag import answer_with_graph, load_graph


def run_query(
    question: str,
    dataset_dir: str = "dataset",
    output_dir: str = "outputs",
    use_openai: bool = False,
) -> None:
    load_dotenv()
    output = Path(output_dir)
    documents = load_documents(dataset_dir)
    retriever = FlatRetriever.load(output / "flat_retriever.pkl")
    graph = load_graph(output / "knowledge_graph.json")

    flat_results = retriever.search(question, top_k=5)
    graph_answer = answer_with_graph(
        graph,
        documents,
        question,
        retriever=retriever,
        use_openai=use_openai,
    )

    print("\nQUESTION")
    print(question)

    print("\nFLAT RAG TOP DOCS")
    for idx, result in enumerate(flat_results, start=1):
        print(f"{idx}. {result.doc_id} | score={result.score:.3f} | {result.title}")
        if result.link:
            print(f"   {result.link}")

    print("\nGRAPH RAG ANSWER")
    print(graph_answer.answer)

    print("\nSEED ENTITIES")
    print(", ".join(graph_answer.seed_entities) or "(none)")

    print("\nGRAPH PATHS")
    for path in graph_answer.paths[:8]:
        print(f"- {path}")

    print("\nEVIDENCE")
    for item in graph_answer.evidence[:5]:
        print(f"- {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the EV corpus with Flat RAG and GraphRAG.")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--use-openai", action="store_true")
    args = parser.parse_args()
    run_query(args.question, args.dataset, args.output, use_openai=args.use_openai)


if __name__ == "__main__":
    main()
