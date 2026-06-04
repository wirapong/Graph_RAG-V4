"""Benchmarking suite (PHASE 8.5).

Run: `python -m metadata_kg.tests.benchmark`

Measures:
- precision / recall on metadata extraction (against a synthetic gold set)
- keyword vs semantic vs hybrid search quality (P@k, MRR)
- writes benchmark_report.json

Designed to run without LLM by using the deterministic extractor.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from metadata_kg.core.kg_builder import MetadataKnowledgeGraph
from metadata_kg.core.llm_agent import MetadataAgent
from metadata_kg.core.metadata_schema import DCATEntityType
from metadata_kg.search.semantic_search import SemanticMetadataSearch

# ---------------------------------------------------------------------------
# Gold set for extraction
# ---------------------------------------------------------------------------
EXTRACTION_GOLD = [
    {
        "text": "Title: COVID-19 Daily Cases Thailand\nDescription: Daily case counts.",
        "expected_keys": {"dct:title", "dct:description"},
        "expected_in_title": "COVID-19",
    },
    {
        "text": "Title: Bangkok Air Quality\nDescription: PM2.5 stations.",
        "expected_keys": {"dct:title", "dct:description"},
        "expected_in_title": "Bangkok",
    },
    {
        "text": "Title: Iris Dataset\nDescription: 150 iris flower samples.",
        "expected_keys": {"dct:title", "dct:description"},
        "expected_in_title": "Iris",
    },
    {
        "text": "Title: Khon Kaen Weather\nDescription: Daily temperature.",
        "expected_keys": {"dct:title", "dct:description"},
        "expected_in_title": "Khon Kaen",
    },
]


# ---------------------------------------------------------------------------
# Search benchmark dataset
# ---------------------------------------------------------------------------
SEARCH_DOCS = [
    ("ds:1", "Thailand COVID-19 Daily Cases", "Daily case counts from Ministry of Public Health.", ["covid", "health"]),
    ("ds:2", "Bangkok Air Quality", "PM2.5 measurements at 50 stations.", ["air-quality", "Bangkok"]),
    ("ds:3", "Coronavirus genome sequences", "SARS-CoV-2 genomic data.", ["genome", "virus"]),
    ("ds:4", "Iris Flower Dataset", "Classic iris classification samples.", ["machine-learning", "biology"]),
    ("ds:5", "Khon Kaen Weather", "Daily temperature humidity rainfall.", ["weather", "Thailand"]),
    ("ds:6", "Thai Stock Market 2024", "Daily SET index closing values.", ["finance", "Thailand"]),
    ("ds:7", "Hospital admissions Bangkok", "PM2.5 vs hospital visit correlation.", ["health", "air-quality"]),
]

# (query, gold_entity_id_substring)
SEARCH_QUERIES = [
    ("covid", "ds:1"),
    ("pm2.5 bangkok", "ds:2"),
    ("coronavirus", "ds:3"),
    ("iris flower", "ds:4"),
    ("khon kaen rainfall", "ds:5"),
    ("stock market", "ds:6"),
    ("hospital air", "ds:7"),
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def precision_recall_extraction(agent: MetadataAgent) -> dict[str, float]:
    correct_title = 0
    correct_keys = 0
    total = len(EXTRACTION_GOLD)
    for case in EXTRACTION_GOLD:
        run = agent.run(case["text"])
        if not run.final_entities:
            continue
        ent = run.final_entities[0]
        title = ent.properties.get("dct:title", "")
        if case["expected_in_title"].lower() in str(title).lower():
            correct_title += 1
        if case["expected_keys"].issubset(set(ent.properties.keys())):
            correct_keys += 1
    return {
        "extraction_title_precision": correct_title / total,
        "extraction_key_coverage": correct_keys / total,
        "n_cases": total,
    }


def _mrr_at_k(rank: int | None, k: int = 5) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


def _precision_at_k(rank: int | None, k: int = 5) -> float:
    return 1.0 if (rank is not None and rank <= k) else 0.0


def search_quality(search: SemanticMetadataSearch, method: str, k: int = 5) -> dict[str, float]:
    p_at_k = []
    mrr = []
    latencies = []
    for query, gold_substr in SEARCH_QUERIES:
        t0 = time.perf_counter()
        if method == "semantic":
            results = search.search(query, top_k=k)
        elif method == "keyword":
            results = search.keyword_fallback(query, top_k=k)
        else:
            results = search.hybrid_search(query, top_k=k)
        latencies.append(time.perf_counter() - t0)
        rank = None
        for i, r in enumerate(results, 1):
            if gold_substr in r.entity_id:
                rank = i
                break
        p_at_k.append(_precision_at_k(rank, k))
        mrr.append(_mrr_at_k(rank, k))
    return {
        f"{method}_P@{k}": sum(p_at_k) / len(p_at_k),
        f"{method}_MRR@{k}": sum(mrr) / len(mrr),
        f"{method}_latency_ms_mean": sum(latencies) * 1000 / len(latencies),
    }


def run_benchmark(output_path: str | Path = "benchmark_report.json") -> dict:
    print("=== Metadata KG benchmark ===")

    # Extraction
    kg = MetadataKnowledgeGraph(name="bench_kg")
    agent = MetadataAgent(kg=kg, use_llm=False)
    ex_metrics = precision_recall_extraction(agent)
    print("Extraction:", ex_metrics)

    # Search
    bench_kg = MetadataKnowledgeGraph(name="search_bench_kg")
    for eid, title, desc, kws in SEARCH_DOCS:
        bench_kg.add_entity(eid, DCATEntityType.Dataset, {
            "dct:title": title, "dct:description": desc, "dcat:keyword": kws,
        })
    search = SemanticMetadataSearch()
    search.index_kg(bench_kg)

    search_metrics: dict = {}
    for method in ("keyword", "semantic", "hybrid"):
        m = search_quality(search, method, k=5)
        search_metrics.update(m)
        print(f"{method}: {m}")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "extraction": ex_metrics,
        "search": search_metrics,
        "config": {
            "llm_used": False,
            "embedding": ("sentence-transformers/all-MiniLM-L6-v2" if search._use_st else "hash-fallback"),
            "n_search_docs": len(SEARCH_DOCS),
            "n_search_queries": len(SEARCH_QUERIES),
        },
    }
    out = Path(output_path)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {out.resolve()}")
    return report


if __name__ == "__main__":
    run_benchmark()
