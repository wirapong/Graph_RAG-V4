"""DCAT2 / DCMI aligned metadata schema (Pydantic v2).

Reference:
- DCAT v2: https://www.w3.org/TR/vocab-dcat-2/
- DCMI Terms: https://www.dublincore.org/specifications/dublin-core/dcmi-terms/
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rdflib import DCAT, DCTERMS, FOAF, Namespace

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------
DCMI = Namespace("http://purl.org/dc/terms/")  # Dublin Core terms
PROV = Namespace("http://www.w3.org/ns/prov#")
SCHEMA = Namespace("https://schema.org/")
MKG = Namespace("https://wirapongc.kku.ac.th/ns/metadata-kg#")  # local extension

NAMESPACES: dict[str, Namespace] = {
    "dcat": DCAT,
    "dct": DCTERMS,
    "dcmi": DCMI,
    "foaf": FOAF,
    "prov": PROV,
    "schema": SCHEMA,
    "mkg": MKG,
}


# ---------------------------------------------------------------------------
# Controlled vocabulary
# ---------------------------------------------------------------------------
class DCATEntityType(str, Enum):
    """DCAT 2 core entity types."""

    Catalog = "dcat:Catalog"
    Dataset = "dcat:Dataset"
    DataService = "dcat:DataService"
    Distribution = "dcat:Distribution"
    CatalogRecord = "dcat:CatalogRecord"


class DCMIType(str, Enum):
    """DCMI Type Vocabulary (12 official terms)."""

    Collection = "dcmitype:Collection"
    Dataset = "dcmitype:Dataset"
    Event = "dcmitype:Event"
    Image = "dcmitype:Image"
    InteractiveResource = "dcmitype:InteractiveResource"
    MovingImage = "dcmitype:MovingImage"
    PhysicalObject = "dcmitype:PhysicalObject"
    Service = "dcmitype:Service"
    Software = "dcmitype:Software"
    Sound = "dcmitype:Sound"
    StillImage = "dcmitype:StillImage"
    Text = "dcmitype:Text"


class DomainExtension(str, Enum):
    """Custom domain extensions."""

    Generic = "generic"
    Finance = "finance"
    Health = "health"
    Education = "education"
    GovernmentOpenData = "gov"


# ---------------------------------------------------------------------------
# Core Pydantic models
# ---------------------------------------------------------------------------
class Agent(BaseModel):
    """foaf:Agent — person or organization."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="URI or identifier")
    name: str
    type: Literal["Person", "Organization", "Software"] = "Person"
    email: str | None = None


class Distribution(BaseModel):
    """dcat:Distribution — a specific representation of a Dataset."""

    model_config = ConfigDict(extra="allow")

    id: str
    title: str | None = None
    description: str | None = None
    access_url: str | None = Field(None, alias="accessURL")
    download_url: str | None = Field(None, alias="downloadURL")
    media_type: str | None = Field(None, alias="mediaType")
    format: str | None = None
    byte_size: int | None = Field(None, alias="byteSize")
    license: str | None = None


class DCATDataset(BaseModel):
    """dcat:Dataset — the main metadata record.

    All field names map directly to DCAT2 / DCMI terms.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Identity
    id: str = Field(..., description="dct:identifier (URI)")
    type: DCATEntityType = DCATEntityType.Dataset

    # Mandatory DCAT fields
    title: str = Field(..., description="dct:title")
    description: str = Field(..., description="dct:description")

    # Recommended DCAT fields
    publisher: Agent | None = Field(None, description="dct:publisher")
    creator: list[Agent] | None = Field(None, description="dct:creator")
    contact_point: Agent | None = Field(None, alias="contactPoint")

    # Dates
    issued: datetime | None = None
    modified: datetime | None = None

    # Classification
    keyword: list[str] = Field(default_factory=list, description="dcat:keyword")
    theme: list[str] = Field(default_factory=list, description="dcat:theme")
    dcmi_type: DCMIType | None = Field(None, description="dct:type from DCMI vocab")
    language: list[str] = Field(default_factory=list, description="dct:language (ISO 639)")

    # Rights / licensing
    license: str | None = Field(None, description="dct:license")
    rights: str | None = Field(None, description="dct:rights")
    access_rights: str | None = Field(None, alias="accessRights")

    # Spatial / temporal coverage
    spatial: str | None = Field(None, description="dct:spatial")
    temporal: str | None = Field(None, description="dct:temporal")

    # Distributions
    distribution: list[Distribution] = Field(default_factory=list)

    # Domain extension
    domain: DomainExtension = DomainExtension.Generic
    domain_properties: dict[str, Any] = Field(default_factory=dict)

    # System metadata
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    extracted_by: str | None = None  # agent_id of LLM extractor
    extraction_timestamp: datetime | None = None

    @field_validator("keyword", "theme", "language", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)


# ---------------------------------------------------------------------------
# Mandatory field sets (used by validate.py and policy.py)
# ---------------------------------------------------------------------------
DCAT_MANDATORY_FIELDS: set[str] = {"id", "title", "description"}
DCAT_RECOMMENDED_FIELDS: set[str] = {
    "publisher",
    "issued",
    "modified",
    "keyword",
    "theme",
    "license",
    "distribution",
}

DCMI_CORE_15: set[str] = {
    "title", "creator", "subject", "description", "publisher",
    "contributor", "date", "type", "format", "identifier",
    "source", "language", "relation", "coverage", "rights",
}


def get_mandatory_fields() -> set[str]:
    """Return DCAT 2 mandatory field names."""
    return DCAT_MANDATORY_FIELDS.copy()


def get_recommended_fields() -> set[str]:
    """Return DCAT 2 recommended field names."""
    return DCAT_RECOMMENDED_FIELDS.copy()


def get_dcmi_core() -> set[str]:
    """Return the DCMI 15 core elements."""
    return DCMI_CORE_15.copy()


__all__ = [
    "Agent",
    "DCAT_MANDATORY_FIELDS",
    "DCAT_RECOMMENDED_FIELDS",
    "DCATDataset",
    "DCATEntityType",
    "DCMIType",
    "DCMI_CORE_15",
    "Distribution",
    "DomainExtension",
    "NAMESPACES",
    "get_dcmi_core",
    "get_mandatory_fields",
    "get_recommended_fields",
]
