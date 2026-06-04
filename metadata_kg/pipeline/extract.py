"""LLM-based entity/metadata extraction (PHASE 4 — Phase 1 enrichment).

Wraps the MetadataAgent to operate on ingested payloads.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from metadata_kg.core.llm_agent import AgentRun, MetadataAgent
from metadata_kg.pipeline.ingest import ingest_any


def extract_from_payload(payload: dict[str, Any], agent: MetadataAgent | None = None) -> AgentRun:
    """Run the metadata agent on an ingested payload.

    Payload format: {"source", "kind", "content"}.
    For non-text content (dict/JSON/YAML), we serialize to a text blob first.
    """
    agent = agent if agent is not None else MetadataAgent()

    content = payload.get("content", "")
    if isinstance(content, dict):
        # Convert dict to a text representation the LLM/heuristic can parse
        lines = []
        for k, v in content.items():
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(x) for x in v)
            lines.append(f"{k}: {v}")
        text = "\n".join(lines)
    elif isinstance(content, list):
        text = "\n".join(str(x) for x in content)
    else:
        text = str(content)

    logger.info(f"Extracting from payload source={payload.get('source')} kind={payload.get('kind')} chars={len(text)}")
    return agent.run(text)


def extract_from_input(
    input: str | dict[str, Any],
    agent: MetadataAgent | None = None,
    *,
    source: str | None = None,
) -> AgentRun:
    """One-stop: ingest_any + extract_from_payload."""
    payload = ingest_any(input, source=source)
    return extract_from_payload(payload, agent=agent)


__all__ = ["extract_from_input", "extract_from_payload"]
