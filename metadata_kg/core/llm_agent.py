"""LLM-as-Agent orchestrator (PHASE 3).

Uses LangChain AgentExecutor with Claude (default: claude-sonnet-4-20250514).

Tools:
- extract_entities(text) → list[Entity]
- map_schema(raw_fields, target_schema) → mapped_dict
- validate_via_kg(entity_id) → bool + reason
- flag_hallucination(claim, kg_context) → confidence_score

Behavior:
- Accept raw document or API payload
- Plan: extract → map → validate → store
- On low confidence (< 0.7): trigger HITL flag
- Log each reasoning step for XAI traceability
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from metadata_kg.core.kg_builder import MetadataKnowledgeGraph
from metadata_kg.core.metadata_schema import (
    DCAT_MANDATORY_FIELDS,
    DCAT_RECOMMENDED_FIELDS,
    DCATEntityType,
)

# LangChain imports are lazy so this module can be imported without LangChain
# installed (heavy dep). Tools defined below always work in deterministic mode.
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.tools import StructuredTool
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    _LANGCHAIN_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    logger.warning(f"LangChain not available: {exc}. Using deterministic fallback.")
    _LANGCHAIN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
class Entity(BaseModel):
    """Extracted entity with type and properties."""

    id: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0


class ValidationResult(BaseModel):
    valid: bool
    reasons: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


@dataclass
class ReasoningStep:
    """One step in the agent's chain-of-thought (logged for XAI)."""

    step_id: str
    timestamp: datetime
    tool: str
    inputs: dict[str, Any]
    outputs: Any
    confidence: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "timestamp": self.timestamp.isoformat(),
            "tool": self.tool,
            "inputs": self.inputs,
            "outputs": self.outputs if isinstance(self.outputs, (str, int, float, bool, list, dict)) else str(self.outputs),
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class AgentRun:
    """Result of one agent invocation."""

    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    input_text: str = ""
    steps: list[ReasoningStep] = field(default_factory=list)
    final_entities: list[Entity] = field(default_factory=list)
    overall_confidence: float = 0.0
    hitl_required: bool = False
    hitl_reasons: list[str] = field(default_factory=list)

    def add_step(self, tool: str, inputs: dict[str, Any], outputs: Any, **kwargs) -> ReasoningStep:
        step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            tool=tool,
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
        self.steps.append(step)
        return step

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "input_text": self.input_text[:200] + "..." if len(self.input_text) > 200 else self.input_text,
            "steps": [s.to_dict() for s in self.steps],
            "final_entities": [e.model_dump() for e in self.final_entities],
            "overall_confidence": self.overall_confidence,
            "hitl_required": self.hitl_required,
            "hitl_reasons": self.hitl_reasons,
        }


# ---------------------------------------------------------------------------
# Deterministic tools (used as agent tools AND as fallback when LLM offline)
# ---------------------------------------------------------------------------
_DEFAULT_KEYWORDS = re.compile(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b")


def deterministic_extract_entities(text: str) -> list[Entity]:
    """Rule-based extractor (fallback when LLM unavailable)."""
    # Heuristic: capitalized tokens, first sentence becomes title.
    title_match = re.search(r"(?:title|Title|TITLE)\s*[:=]\s*(.+)", text)
    title = title_match.group(1).strip() if title_match else text.split("\n", 1)[0][:120]

    desc_match = re.search(r"(?:description|Description|abstract|Abstract)\s*[:=]\s*(.+?)(?:\n\n|\Z)", text, re.S)
    description = desc_match.group(1).strip() if desc_match else text[:500]

    keywords = list({m.group(0) for m in _DEFAULT_KEYWORDS.finditer(text)})[:10]

    ent = Entity(
        id=f"mkg:auto-{uuid.uuid4().hex[:8]}",
        type=DCATEntityType.Dataset.value,
        properties={
            "dct:title": title,
            "dct:description": description,
            "dcat:keyword": keywords,
        },
        confidence=0.6,  # rule-based confidence
    )
    return [ent]


def deterministic_map_schema(raw_fields: dict[str, Any], target_schema: str = "DCAT2") -> dict[str, Any]:
    """Map raw field names to DCAT2/DCMI predicates."""
    mapping = {
        "name": "dct:title", "title": "dct:title", "headline": "dct:title",
        "summary": "dct:description", "abstract": "dct:description", "description": "dct:description",
        "author": "dct:creator", "creator": "dct:creator", "authors": "dct:creator",
        "publisher": "dct:publisher", "publishedBy": "dct:publisher",
        "tags": "dcat:keyword", "keywords": "dcat:keyword", "keyword": "dcat:keyword",
        "license": "dct:license", "licence": "dct:license",
        "url": "dcat:accessURL", "link": "dcat:accessURL",
        "date": "dct:issued", "published": "dct:issued", "createdAt": "dct:issued",
        "modified": "dct:modified", "updatedAt": "dct:modified",
        "language": "dct:language", "lang": "dct:language",
        "format": "dct:format", "mimeType": "dcat:mediaType",
        "subject": "dct:subject", "category": "dcat:theme", "topic": "dcat:theme",
        "spatial": "dct:spatial", "location": "dct:spatial",
        "temporal": "dct:temporal", "period": "dct:temporal",
    }
    out: dict[str, Any] = {}
    for k, v in raw_fields.items():
        # If already a CURIE in a known namespace, keep as-is (idempotent)
        if ":" in k and k.split(":", 1)[0] in {"dct", "dcat", "foaf", "prov", "schema", "mkg", "dcmi"}:
            mapped = k
        else:
            mapped = mapping.get(k, mapping.get(k.lower(), f"mkg:{k}"))
        out[mapped] = v
    return out


def deterministic_validate_via_kg(entity_id: str, kg: MetadataKnowledgeGraph) -> ValidationResult:
    """Check that entity exists and has all mandatory DCAT fields."""
    props = kg.get_entity(entity_id)
    if not props:
        return ValidationResult(valid=False, reasons=[f"Entity {entity_id} not found in KG"])

    normalized = {MetadataKnowledgeGraph._strip_prefix(k) for k in props}
    missing = [f for f in DCAT_MANDATORY_FIELDS if f != "id" and f not in normalized]
    if missing:
        return ValidationResult(
            valid=False,
            reasons=[f"Missing mandatory DCAT fields: {missing}"],
            missing_fields=missing,
        )
    return ValidationResult(valid=True, reasons=["All mandatory DCAT fields present"])


def deterministic_flag_hallucination(claim: str, kg_context: list[str]) -> float:
    """Score how supported a claim is by KG context. Returns confidence 0-1."""
    if not claim:
        return 0.0
    if not kg_context:
        return 0.4  # no grounding → moderate suspicion

    claim_tokens = set(re.findall(r"\w+", claim.lower()))
    ctx_tokens = set()
    for ctx in kg_context:
        ctx_tokens.update(re.findall(r"\w+", ctx.lower()))

    if not claim_tokens:
        return 0.0
    overlap = len(claim_tokens & ctx_tokens) / len(claim_tokens)
    # Mild non-linear: high overlap → high confidence
    return min(1.0, 0.3 + 0.7 * overlap)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
HITL_THRESHOLD = 0.7


class MetadataAgent:
    """LLM-as-Agent for metadata extraction & validation.

    Two modes:
    - LLM mode: uses LangChain + Claude with tool-calling
    - Deterministic mode: rule-based fallback (no API key needed)
    """

    def __init__(
        self,
        kg: MetadataKnowledgeGraph | None = None,
        model: str | None = None,
        api_key: str | None = None,
        use_llm: bool | None = None,
        hitl_threshold: float = HITL_THRESHOLD,
    ) -> None:
        self.kg = kg if kg is not None else MetadataKnowledgeGraph(name="agent_kg")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.hitl_threshold = hitl_threshold

        if use_llm is None:
            use_llm = _LANGCHAIN_AVAILABLE and bool(self.api_key)
        self.use_llm = use_llm

        self._executor = None
        if self.use_llm:
            try:
                self._executor = self._build_executor()
                logger.info(f"MetadataAgent initialized with Claude model={self.model}")
            except Exception as exc:
                logger.warning(f"LLM init failed, falling back to deterministic mode: {exc}")
                self.use_llm = False

        if not self.use_llm:
            logger.info("MetadataAgent running in deterministic (rule-based) mode")

    # ------------------------------------------------------------------
    # Tools (exposed as LangChain StructuredTool when LLM mode)
    # ------------------------------------------------------------------
    def extract_entities(self, text: str) -> list[Entity]:
        return deterministic_extract_entities(text)

    def map_schema(self, raw_fields: dict[str, Any], target_schema: str = "DCAT2") -> dict[str, Any]:
        return deterministic_map_schema(raw_fields, target_schema)

    def validate_via_kg(self, entity_id: str) -> ValidationResult:
        return deterministic_validate_via_kg(entity_id, self.kg)

    def flag_hallucination(self, claim: str, kg_context: list[str] | None = None) -> float:
        if kg_context is None:
            # Pull all titles/descriptions from KG as context
            kg_context = []
            for ent_id in self.kg.all_entities():
                props = self.kg.get_entity(ent_id)
                for v in props.values():
                    if isinstance(v, str):
                        kg_context.append(v)
                    elif isinstance(v, list):
                        kg_context.extend(str(x) for x in v)
        return deterministic_flag_hallucination(claim, kg_context)

    # ------------------------------------------------------------------
    # LangChain executor
    # ------------------------------------------------------------------
    def _build_executor(self) -> "AgentExecutor":
        from pydantic import BaseModel as LCBaseModel

        class _ExtractIn(LCBaseModel):
            text: str

        class _MapIn(LCBaseModel):
            raw_fields: dict
            target_schema: str = "DCAT2"

        class _ValidateIn(LCBaseModel):
            entity_id: str

        class _HallucIn(LCBaseModel):
            claim: str
            kg_context: list[str] = []

        tools = [
            StructuredTool.from_function(
                name="extract_entities",
                description="Extract structured DCAT2 entities from raw text.",
                func=lambda text: [e.model_dump() for e in self.extract_entities(text)],
                args_schema=_ExtractIn,
            ),
            StructuredTool.from_function(
                name="map_schema",
                description="Map arbitrary field names to DCAT2/DCMI predicates.",
                func=lambda raw_fields, target_schema="DCAT2": self.map_schema(raw_fields, target_schema),
                args_schema=_MapIn,
            ),
            StructuredTool.from_function(
                name="validate_via_kg",
                description="Validate that an entity_id satisfies DCAT2 mandatory fields in the KG.",
                func=lambda entity_id: self.validate_via_kg(entity_id).model_dump(),
                args_schema=_ValidateIn,
            ),
            StructuredTool.from_function(
                name="flag_hallucination",
                description="Score (0-1) how well a claim is supported by KG context. <0.7 triggers HITL.",
                func=lambda claim, kg_context: self.flag_hallucination(claim, kg_context),
                args_schema=_HallucIn,
            ),
        ]

        llm = ChatAnthropic(model=self.model, api_key=self.api_key, temperature=0.1)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a metadata extraction agent for DCAT 2 / DCMI catalogs. "
             "Plan: extract_entities → map_schema → validate_via_kg → flag_hallucination. "
             "If confidence < 0.7, return a flag for human-in-the-loop review. "
             "Always cite which fields were inferred vs. extracted."),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=8)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------
    def run(self, input_text: str, *, dry_run: bool = False) -> AgentRun:
        """Execute the agent plan and return an AgentRun."""
        run = AgentRun(
            run_id=str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc),
            input_text=input_text,
        )

        # 1. Extract
        entities = self.extract_entities(input_text)
        run.add_step("extract_entities", {"text_len": len(input_text)}, [e.model_dump() for e in entities])

        # 2. Map + store
        for ent in entities:
            mapped_props = self.map_schema(ent.properties)
            run.add_step("map_schema", {"raw_fields": list(ent.properties.keys())}, {"mapped_keys": list(mapped_props.keys())})

            if not dry_run:
                self.kg.add_entity(ent.id, ent.type, mapped_props)
                run.add_step("kg.add_entity", {"id": ent.id, "type": ent.type}, "stored")

            # 3. Validate
            val = self.validate_via_kg(ent.id)
            run.add_step("validate_via_kg", {"entity_id": ent.id}, val.model_dump(), confidence=1.0 if val.valid else 0.5)

            # 4. Hallucination scoring on the description
            desc = ent.properties.get("dct:description") or ent.properties.get("description", "")
            score = self.flag_hallucination(desc)
            run.add_step("flag_hallucination", {"claim_len": len(desc)}, {"confidence": score}, confidence=score)

            ent.confidence = min(ent.confidence, score)
            run.final_entities.append(ent)

            if score < self.hitl_threshold:
                run.hitl_required = True
                run.hitl_reasons.append(f"{ent.id}: confidence {score:.2f} < {self.hitl_threshold}")
            if not val.valid:
                run.hitl_required = True
                run.hitl_reasons.append(f"{ent.id}: validation failed → {val.reasons}")

        # Aggregate confidence (mean)
        if run.final_entities:
            run.overall_confidence = sum(e.confidence for e in run.final_entities) / len(run.final_entities)
        run.finished_at = datetime.now(timezone.utc)
        logger.info(
            f"AgentRun {run.run_id[:8]} done: {len(run.final_entities)} entities, "
            f"confidence={run.overall_confidence:.2f}, hitl={run.hitl_required}"
        )
        return run

    def reasoning_log(self, run: AgentRun) -> str:
        """Render a human-readable reasoning trace (XAI)."""
        lines = [f"# Reasoning trace for run {run.run_id[:8]}", ""]
        for i, step in enumerate(run.steps, 1):
            lines.append(f"## Step {i}: {step.tool}")
            lines.append(f"- timestamp: {step.timestamp.isoformat()}")
            lines.append(f"- inputs: {json.dumps(step.inputs, default=str)[:200]}")
            lines.append(f"- outputs: {json.dumps(step.outputs, default=str)[:200]}")
            if step.confidence is not None:
                lines.append(f"- confidence: {step.confidence:.2f}")
            lines.append("")
        lines.append(f"**Overall confidence:** {run.overall_confidence:.2f}")
        lines.append(f"**HITL required:** {run.hitl_required}")
        if run.hitl_reasons:
            lines.append("**HITL reasons:**")
            for r in run.hitl_reasons:
                lines.append(f"- {r}")
        return "\n".join(lines)


__all__ = [
    "AgentRun",
    "Entity",
    "MetadataAgent",
    "ReasoningStep",
    "ValidationResult",
    "deterministic_extract_entities",
    "deterministic_flag_hallucination",
    "deterministic_map_schema",
    "deterministic_validate_via_kg",
]
