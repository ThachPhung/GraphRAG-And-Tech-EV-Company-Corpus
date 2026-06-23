from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path

import networkx as nx
from openai import OpenAI

from data_loader import Document, split_sentences
from entity_extraction import Triple, extract_entities
from flat_rag import FlatRetriever


@dataclass(frozen=True)
class GraphAnswer:
    answer: str
    seed_entities: list[str]
    evidence: list[str]
    paths: list[str]


def infer_node_type(name: str) -> str:
    lower = name.lower()
    if any(place in lower for place in ["united states", "california", "china", "europe", "north america"]):
        return "LOCATION"
    if any(term in lower for term in ["sales", "market", "sentiment", "financial"]):
        return "TOPIC"
    if any(term in lower for term in ["act", "regulation", "policy", "incentive"]):
        return "POLICY"
    if any(term in lower for term in ["battery", "charging", "hydrogen", "fuel cell"]):
        return "TECHNOLOGY"
    return "ORGANIZATION_OR_ENTITY"


def build_graph(triples: list[Triple]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for triple in triples:
        for node in [triple.subject, triple.object]:
            if not graph.has_node(node):
                graph.add_node(node, label=node, type=infer_node_type(node))

        graph.add_edge(
            triple.subject,
            triple.object,
            key=f"{triple.predicate}:{triple.doc_id}:{len(graph.edges)}",
            predicate=triple.predicate,
            doc_id=triple.doc_id,
            evidence=triple.evidence,
            confidence=triple.confidence,
        )
    return graph


def save_graph(graph: nx.MultiDiGraph, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(graph, output / "knowledge_graph.gexf")
    data = nx.node_link_data(graph, edges="edges")
    (output / "knowledge_graph.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_graph(path: str | Path) -> nx.MultiDiGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return nx.node_link_graph(data, edges="edges")


def export_neo4j_csv(graph: nx.MultiDiGraph, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    nodes = ["id:ID,label,type:LABEL"]
    for node, attrs in graph.nodes(data=True):
        safe = str(node).replace('"', "'")
        nodes.append(f'"{safe}","{safe}",{attrs.get("type", "Entity")}')

    rels = [":START_ID,:END_ID,:TYPE,doc_id,evidence"]
    for src, dst, attrs in graph.edges(data=True):
        safe_src = str(src).replace('"', "'")
        safe_dst = str(dst).replace('"', "'")
        evidence = str(attrs.get("evidence", "")).replace('"', "'")[:500]
        rels.append(
            f'"{safe_src}","{safe_dst}",{attrs.get("predicate", "RELATED_TO")},"{attrs.get("doc_id", "")}","{evidence}"'
        )

    (output / "neo4j_nodes.csv").write_text("\n".join(nodes), encoding="utf-8")
    (output / "neo4j_relationships.csv").write_text("\n".join(rels), encoding="utf-8")


def graph_context(
    graph: nx.MultiDiGraph,
    question: str,
    retriever: FlatRetriever | None = None,
    radius: int = 2,
    top_k: int = 12,
) -> tuple[list[str], list[str], list[str]]:
    seeds = [entity for entity in extract_entities(question) if graph.has_node(entity)]
    lower_question = question.lower()
    if any(term in lower_question for term in ["incentive", "policy", "regulation", "tax credit"]):
        seeds.extend(
            seed
            for seed in ["Inflation Reduction Act", "California Air Resources Board", "CARB", "ZEV", "California"]
            if graph.has_node(seed)
        )
    if any(term in lower_question for term in ["charging", "charger", "infrastructure"]):
        seeds.extend(seed for seed in ["McKinsey", "United States"] if graph.has_node(seed))
    seeds = list(dict.fromkeys(seeds))[:6]
    top_doc_ids: set[str] = set()
    if retriever:
        top_doc_ids = {result.doc_id for result in retriever.search(question, top_k=5)}

    if not seeds and retriever:
        for result in retriever.search(question, top_k=3):
            for entity in extract_entities(result.text):
                if graph.has_node(entity):
                    seeds.append(entity)
        seeds = list(dict.fromkeys(seeds))[:5]

    evidence_scores: dict[str, int] = {}
    paths: list[str] = []
    seen_evidence: set[str] = set()
    question_terms = {token.lower() for token in question.split() if len(token) > 3}

    for seed in seeds:
        neighborhood = nx.ego_graph(graph.to_undirected(), seed, radius=radius).nodes
        for src, dst, attrs in graph.edges(data=True):
            if src not in neighborhood or dst not in neighborhood:
                continue
            predicate = attrs.get("predicate", "RELATED_TO")
            doc_id = attrs.get("doc_id", "")
            text = attrs.get("evidence", "")
            paths.append(f"{src} -[{predicate}]-> {dst} ({doc_id})")
            if text and len(text) >= 60 and text not in seen_evidence:
                score = sum(1 for term in question_terms if term in text.lower())
                if doc_id in top_doc_ids:
                    score += 5
                if any(term in lower_question for term in ["incentive", "policy", "regulation", "tax credit"]):
                    if predicate in {"REGULATED_OR_REQUIRED_BY", "SUPPORTED_BY_INCENTIVE"}:
                        score += 10
                if score == 0:
                    continue
                evidence_scores[text] = score
                seen_evidence.add(text)

    evidence = [
        text
        for text, _ in sorted(
            evidence_scores.items(),
            key=lambda item: (item[1], len(item[0])),
            reverse=True,
        )
    ]
    return seeds, evidence[:top_k], list(dict.fromkeys(paths))[:top_k]


def answer_with_graph(
    graph: nx.MultiDiGraph,
    documents: list[Document],
    question: str,
    retriever: FlatRetriever | None = None,
    use_openai: bool = False,
) -> GraphAnswer:
    seeds, evidence, paths = graph_context(graph, question, retriever=retriever)
    if not evidence and retriever:
        evidence = [result.text for result in retriever.search(question, top_k=3)]

    if use_openai and os.getenv("OPENAI_API_KEY"):
        answer = _llm_answer(question, evidence, paths)
    else:
        answer = _offline_answer(question, evidence, paths)
    return GraphAnswer(answer=answer, seed_entities=seeds, evidence=evidence, paths=paths)


def _llm_answer(question: str, evidence: list[str], paths: list[str]) -> str:
    client = OpenAI()
    context = "\n".join(f"- {item}" for item in evidence[:10])
    graph_paths = "\n".join(f"- {path}" for path in paths[:10])
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "Answer using only the supplied evidence. Be concise and cite graph facts in plain language.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nEvidence:\n{context}\n\nGraph paths:\n{graph_paths}",
            },
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def _offline_answer(question: str, evidence: list[str], paths: list[str]) -> str:
    if not evidence:
        return "Không tìm thấy evidence đủ mạnh trong dataset cho câu hỏi này."

    top: list[str] = []
    for item in evidence:
        for sentence in split_sentences(item) or [item]:
            if sentence not in top:
                top.append(sentence)
            if len(top) >= 4:
                break
        if len(top) >= 4:
            break

    path_summary = ""
    if paths:
        predicates = Counter(path.split("-[", 1)[1].split("]", 1)[0] for path in paths if "-[" in path)
        labels = ", ".join(f"{name} ({count})" for name, count in predicates.most_common(3))
        path_summary = f"\n\nGraph signals: {labels}."

    return " ".join(top) + path_summary
