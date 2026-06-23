from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
import re
from typing import Iterable

from openai import OpenAI

from data_loader import Document, split_sentences


OPENAI_USAGE = {
    "requests": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "failed_requests": 0,
    "fallback_documents": 0,
}


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    doc_id: str
    evidence: str
    confidence: float = 0.7

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


KNOWN_ENTITIES = [
    "Tesla",
    "Ford",
    "General Motors",
    "GM",
    "Chevrolet",
    "Chevy Bolt",
    "Rivian",
    "Lucid",
    "Nikola",
    "Hyundai",
    "Kia",
    "BMW",
    "Mercedes",
    "Mercedes-Benz",
    "Cadillac",
    "Vinfast",
    "Audi",
    "Toyota",
    "Honda",
    "Volkswagen",
    "Cox Automotive",
    "Kelley Blue Book",
    "McKinsey",
    "BloombergNEF",
    "BNEF",
    "ICCT",
    "Inflation Reduction Act",
    "California Air Resources Board",
    "CARB",
    "California",
    "United States",
    "U.S.",
    "North America",
    "China",
    "Europe",
    "India",
    "Thailand",
    "Brazil",
    "Phoenix",
    "Coolidge",
    "HYLA",
    "Voltera",
    "Iveco",
    "Walmart",
    "Linde",
    "Biagi Bros",
]

ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+){0,4}|[A-Z]{2,})\b"
)

STOP_ENTITIES = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "which",
    "why",
    "full content",
    "query",
    "title",
    "link",
    "snippet",
}


def normalize_entity(entity: str) -> str:
    entity = re.sub(r"\s+", " ", entity).strip(" ,.;:()[]{}\"'")
    if entity.startswith("US ") or entity.startswith("U.S. "):
        return "United States"
    aliases = {
        "US": "United States",
        "U.S": "United States",
        "U.S.": "United States",
        "GM": "General Motors",
        "KBB": "Kelley Blue Book",
        "BNEF": "BloombergNEF",
        "Mercedes": "Mercedes-Benz",
    }
    return aliases.get(entity, entity)


def extract_entities(text: str) -> list[str]:
    found: set[str] = set()
    lower = text.lower()
    for entity in KNOWN_ENTITIES:
        if entity.lower() in lower:
            found.add(normalize_entity(entity))

    for candidate in ENTITY_PATTERN.findall(text):
        candidate = normalize_entity(candidate)
        if len(candidate) < 3:
            continue
        if candidate.lower() in STOP_ENTITIES:
            continue
        if len(candidate.split()) == 1 and candidate not in KNOWN_ENTITIES and not candidate.isupper():
            continue
        found.add(candidate)

    return sorted(found)


def infer_predicate(sentence: str) -> str:
    s = sentence.lower()
    rules = [
        (("founded", "established"), "FOUNDED_OR_ESTABLISHED"),
        (("reported", "announced", "said"), "REPORTED"),
        (("forecast", "expects", "expected", "projects"), "FORECASTS"),
        (("increase", "grew", "growth", "rose", "up"), "GREW_OR_INCREASED"),
        (("decline", "decrease", "fell", "down", "slowed", "lower"), "DECLINED_OR_SLOWED"),
        (("invest", "investment", "spending"), "INVESTS_IN"),
        (("partner", "joint development", "collaboration"), "PARTNERED_WITH"),
        (("incentive", "tax credit", "inflation reduction act"), "SUPPORTED_BY_INCENTIVE"),
        (("regulation", "mandating", "require", "policy"), "REGULATED_OR_REQUIRED_BY"),
        (("charging", "charger", "infrastructure", "ports"), "RELATED_TO_CHARGING"),
        (("battery", "range"), "RELATED_TO_BATTERY"),
        (("hydrogen", "fuel cell"), "RELATED_TO_HYDROGEN"),
        (("market share", "share"), "HAS_MARKET_SHARE"),
        (("sales", "deliveries", "volume"), "RELATED_TO_SALES"),
    ]
    for needles, predicate in rules:
        if any(needle in s for needle in needles):
            return predicate
    return "RELATED_TO"


def extract_rule_based_triples(document: Document, max_sentences: int = 80) -> list[Triple]:
    triples: list[Triple] = []
    sentences = split_sentences(document.searchable_text)[:max_sentences]

    for sentence in sentences:
        entities = extract_entities(sentence)
        if len(entities) < 2:
            continue
        predicate = infer_predicate(sentence)
        for subject, obj in _entity_pairs(entities):
            triples.append(
                Triple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    doc_id=document.doc_id,
                    evidence=sentence[:700],
                    confidence=0.65,
                )
            )

    # Link each document to its query topic so browsing starts from useful anchors.
    if document.query:
        triples.append(
            Triple(
                subject=document.title or document.doc_id,
                predicate="MATCHES_SEARCH_QUERY",
                object=document.query,
                doc_id=document.doc_id,
                evidence=document.snippet or document.title,
                confidence=0.8,
            )
        )
    return triples


def _entity_pairs(entities: Iterable[str], limit: int = 4) -> list[tuple[str, str]]:
    unique = list(dict.fromkeys(entities))[:limit]
    return [
        (unique[i], unique[j])
        for i in range(len(unique))
        for j in range(i + 1, len(unique))
        if unique[i] != unique[j]
    ]


def extract_openai_triples(document: Document, model: str | None = None) -> list[Triple]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required when --use-openai is enabled.")

    client = OpenAI()
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    text = document.searchable_text[:6000]
    prompt = f"""
Extract knowledge graph triples from this EV-sector document.
Return JSON only as an object with key "triples".
Each triple must have keys: subject, predicate, object, evidence, confidence.
Use concise uppercase predicates like FOUNDED_BY, REPORTED_SALES, PARTNERED_WITH, REGULATED_BY.
Extract 5-12 high-value triples. Prefer companies, policies, technologies, locations, metrics, and market events.

Document ID: {document.doc_id}
Text:
{text}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    OPENAI_USAGE["requests"] += 1
    if response.usage:
        OPENAI_USAGE["prompt_tokens"] += response.usage.prompt_tokens
        OPENAI_USAGE["completion_tokens"] += response.usage.completion_tokens
        OPENAI_USAGE["total_tokens"] += response.usage.total_tokens

    content = response.choices[0].message.content or "[]"
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    data = json.loads(content)
    if isinstance(data, dict):
        data = data.get("triples", [])
    triples = []
    for row in data:
        triples.append(
            Triple(
                subject=normalize_entity(str(row["subject"])),
                predicate=str(row["predicate"]).upper().replace(" ", "_"),
                object=normalize_entity(str(row["object"])),
                doc_id=document.doc_id,
                evidence=str(row.get("evidence", ""))[:700],
                confidence=float(row.get("confidence", 0.85)),
            )
        )
    return triples


def extract_triples(document: Document, use_openai: bool = False) -> list[Triple]:
    if use_openai:
        try:
            triples = extract_openai_triples(document)
            if triples:
                return triples
        except Exception as exc:
            OPENAI_USAGE["failed_requests"] += 1
            print(f"[WARN] OpenAI extraction failed for {document.doc_id}: {exc}")
        OPENAI_USAGE["fallback_documents"] += 1
    return extract_rule_based_triples(document)
