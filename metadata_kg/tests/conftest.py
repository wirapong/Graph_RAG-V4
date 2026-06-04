"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from metadata_kg.core.kg_builder import MetadataKnowledgeGraph
from metadata_kg.core.llm_agent import MetadataAgent
from metadata_kg.core.metadata_schema import DCATEntityType
from metadata_kg.governance.lineage import DataLineage
from metadata_kg.pipeline.lifecycle import MetadataLifecycle


@pytest.fixture
def empty_kg() -> MetadataKnowledgeGraph:
    return MetadataKnowledgeGraph(name="test_kg")


@pytest.fixture
def sample_kg() -> MetadataKnowledgeGraph:
    kg = MetadataKnowledgeGraph(name="sample_kg")
    kg.add_entity(
        "ds:001",
        DCATEntityType.Dataset,
        {
            "dct:title": "Thailand COVID-19 Daily Cases",
            "dct:description": "Daily case counts from Ministry of Public Health.",
            "dcat:keyword": ["covid", "health", "Thailand"],
            "dct:license": "CC-BY-4.0",
        },
    )
    kg.add_entity(
        "ds:002",
        DCATEntityType.Dataset,
        {
            "dct:title": "Bangkok Air Quality 2024",
            "dct:description": "PM2.5 measurements at 50 stations.",
            "dcat:keyword": ["air-quality", "Bangkok"],
        },
    )
    return kg


@pytest.fixture
def agent_no_llm(empty_kg: MetadataKnowledgeGraph) -> MetadataAgent:
    return MetadataAgent(kg=empty_kg, use_llm=False)


@pytest.fixture
def lineage() -> DataLineage:
    return DataLineage()


@pytest.fixture
def lifecycle(empty_kg, lineage):
    agent = MetadataAgent(kg=empty_kg, use_llm=False)
    return MetadataLifecycle(kg=empty_kg, agent=agent, lineage=lineage)
