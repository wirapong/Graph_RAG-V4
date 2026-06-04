<div align="center">

# 🕸️ Metadata KG

### AI Metadata Automation with Knowledge Graphs

**DCAT 2 / DCMI–aligned · LLM-as-Agent · XAI · Human-in-the-Loop**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-46%20passed-brightgreen.svg)](#-testing)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![rdflib](https://img.shields.io/badge/rdflib-7.6-orange)](https://rdflib.readthedocs.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-purple)](https://www.langchain.com)
[![Claude](https://img.shields.io/badge/Claude-sonnet--4-D97757?logo=anthropic&logoColor=white)](https://www.anthropic.com)

[**Quick start**](#-quick-start) · [**Live demo**](#-deploy-to-streamlit-cloud) · [**Docs**](#-documentation) · [**API**](#-fastapi-server) · [**Cite**](#-citation)

</div>

---

## 📖 Table of Contents

- [Why Metadata KG?](#-why-metadata-kg)
- [Features](#-features)
- [Architecture](#%EF%B8%8F-architecture)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [Python API](#1-python-api)
  - [FastAPI server](#2-fastapi-server)
  - [Streamlit UI](#3-streamlit-ui)
- [Deploy to Streamlit Cloud](#-deploy-to-streamlit-cloud)
- [HITL Workflow](#-human-in-the-loop-workflow)
- [Project Layout](#-project-layout)
- [Configuration](#%EF%B8%8F-configuration)
- [Benchmarks](#-benchmarks)
- [Testing](#-testing)
- [Standards](#-standards--references)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## 💡 Why Metadata KG?

Modern data catalogues (DCAT 2 / DCMI) need three things at once — **machine-readable metadata**, **provenance** for every claim, and a **human review path** when an AI guess looks shaky.

Metadata KG ships all three in one Python package:

- An **LLM-as-Agent** (Claude `sonnet-4`) plans the extraction pipeline and exposes 4 callable tools.
- A **dual-store Knowledge Graph** (rdflib + networkx) gives you Turtle/SPARQL on one side and `networkx` analytics + GNN-ready graphs on the other.
- **Lineage** (PROV-O JSON-LD), **policy-as-code** (GDPR PII, mandatory DCAT fields), and **XAI** are first-class — not bolted on.
- Everything runs **offline** with a deterministic fallback when no API key is set.

Built for academic catalogues, gov-open-data portals, and internal data products.

---

## ✨ Features

| Category | What you get |
|----------|--------------|
| 🧠 **LLM-as-Agent** | Claude `sonnet-4` via LangChain · 4 tools (`extract_entities`, `map_schema`, `validate_via_kg`, `flag_hallucination`) · auto-plan → extract → map → validate → store |
| 🕸️ **Knowledge Graph** | rdflib (Turtle / SPARQL-ready) + networkx (graph algos / GNN export) · DCAT 2 schema · cross-domain linking via `mkg:sameAs` |
| ♻️ **Lifecycle** | 4 phases: Creation → Cleaning (Jaccard dedup) → Maintenance (versioned updates) → Retirement (archive / GDPR purge) |
| 🛡️ **Governance** | PROV-O lineage · YAML-loadable policy rules · GDPR PII regex (email, phone, Thai national ID, credit card, US SSN) · DCAT mandatory-field enforcement |
| 🧩 **XAI** | Per-entity Markdown explanation · per-field confidence report · full reasoning trace |
| 🔎 **Search** | Sentence-Transformers (with hash-based fallback) · BM25 · hybrid · query expansion · result explanations |
| 🌐 **REST API** | 12 FastAPI endpoints · standard envelope response · OpenAPI/Swagger UI |
| 👥 **HITL** | Auto-flag low confidence (<0.7) · review queue · versioned corrections |
| 🎨 **Streamlit UI** | 5-tab dashboard · ingest, search, explain, govern, graph · Streamlit Cloud–ready |
| 🧪 **Quality** | 46 pytest tests · benchmark suite · ruff/mypy config |

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Raw document<br/>text · JSON · YAML · PDF] --> B[Ingest]
    B --> C[LLM-as-Agent<br/>extract → map → validate]
    C --> D[(Knowledge Graph<br/>rdflib + networkx)]
    D --> E[Validate &<br/>Deduplicate]
    E --> F{Confidence < 0.7?}
    F -->|yes| H[HITL Queue]
    F -->|no| G[Search Index]
    H --> I[Human Review]
    I --> D
    G --> J[Semantic / BM25 / Hybrid Search]
    D --> K[XAI Explanation]
    D --> L[Lineage JSON-LD]
    D --> M[Turtle Export]
    subgraph Governance
        N[PolicyEngine<br/>GDPR · DCAT 2]
    end
    D --> N
```

> See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the layered diagram and per-module responsibilities.

---

## 🚀 Quick Start

### Prerequisites

- Python **3.11+** (tested on 3.11.15)
- ~500 MB free disk (for optional `sentence-transformers` model)
- _Optional_: an [Anthropic API key](https://console.anthropic.com) to enable LLM mode

### Install

```bash
git clone https://github.com/<your-username>/metadata-kg.git
cd metadata-kg

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"            # editable + dev tools
```

For PDF ingestion: `pip install -e ".[dev,pdf]"`.

### Configure (optional)

```bash
cp .env.example .env
# Edit .env to add ANTHROPIC_API_KEY=sk-ant-…
# Without a key, the agent uses the deterministic fallback.
```

### Run the quickstart

```bash
python examples/quickstart.py
```

You should see something like:

```
=== Metadata KG quickstart ===
✅ Created entity: mkg:auto-63773385
🔎 Search 'air quality':
   0.600  …auto-63773385  Thailand Air Quality 2024 ...
🛡️ Policy: pass=False, errors=1
   - [error] CUSTOM_LICENSE_REQUIRED: Mandatory field 'license' missing
🧠 XAI explanation (first 400 chars): ...
💾 Saved: data/quickstart.ttl + quickstart_lineage.jsonld
```

---

## 🧰 Usage

### 1. Python API

```python
from metadata_kg.pipeline.lifecycle import MetadataLifecycle

lc = MetadataLifecycle()                       # KG + Lineage + Agent
results = lc.run_full(
    "Title: Bangkok PM2.5 2024\n"
    "Description: Hourly PM2.5 measurements from 50 stations.",
    source="catalog/manual",
)
eid = results["creation"].entity_ids[0]
print(lc.kg.get_entity(eid))
print(lc.kg.export_to_turtle())
```

### 2. FastAPI server

```bash
metadata-kg-api                    # or: python -m metadata_kg.api.routes
# → http://localhost:8000/docs (Swagger UI)
```

<details>
<summary><b>Endpoint reference (click to expand)</b></summary>

| Method | Path                    | Purpose                                  |
|--------|-------------------------|------------------------------------------|
| `POST` | `/ingest`               | Submit raw text → extract + store        |
| `GET`  | `/metadata/{id}`        | Retrieve metadata + provenance           |
| `GET`  | `/search?q=…&method=…`  | Semantic / keyword / hybrid search       |
| `POST` | `/validate/{id}`        | DCAT validation report                   |
| `GET`  | `/explain/{id}`         | XAI human-readable explanation           |
| `POST` | `/hitl/review`          | Submit human correction/approval         |
| `GET`  | `/hitl/queue`           | Items flagged for review                 |
| `POST` | `/policy/check`         | Run policy compliance on any dict        |
| `GET`  | `/graph/turtle`         | Export full KG as Turtle                 |
| `GET`  | `/graph/lineage`        | Export lineage as PROV-O JSON-LD         |
| `GET`  | `/graph/validate`       | Full-graph consistency check             |
| `GET`  | `/stats`                | KG + lineage stats                       |

All responses share an envelope:

```json
{
  "data": { ... },
  "confidence": 0.85,
  "lineage_url": "/explain/<entity_id>",
  "warnings": [],
  "timestamp": "2026-06-04T08:25:00+00:00"
}
```

</details>

### 3. Streamlit UI

```bash
streamlit run streamlit_app.py
# → http://localhost:8501
```

Five tabs: **Ingest · Search · Explain · Governance · Graph** — each driven by the same Python API.

> _Screenshots coming soon — drop your own under `docs/screenshots/` and embed them here._

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to <https://streamlit.io/cloud> → **New app**.
3. Repository: `<you>/metadata-kg` · Main file: `streamlit_app.py` · Python: **3.11**.
4. Add **Secrets** (Streamlit dashboard → ⚙️ → Secrets):
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-xxxxxxxxxxxxxxxxxxxx"
   ANTHROPIC_MODEL   = "claude-sonnet-4-20250514"
   ```
5. Click **Deploy**. `requirements.txt` is picked up automatically.

> 💡 The first run downloads `sentence-transformers/all-MiniLM-L6-v2` (~80 MB). If memory is tight, the app falls back to a 256-dim hashed bag-of-words automatically.

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for Docker, server, and CI options.

---

## 👥 Human-in-the-Loop Workflow

```mermaid
sequenceDiagram
    actor User
    participant API
    participant Agent
    participant KG
    participant Reviewer
    User->>API: POST /ingest (text)
    API->>Agent: run()
    Agent->>Agent: extract → map → validate → score
    alt confidence ≥ 0.7
        Agent->>KG: add_entity
        API-->>User: 200 (metadata, lineage)
    else confidence < 0.7
        Agent->>KG: add_entity (flagged)
        Agent->>API: hitl_required=true
        API-->>Reviewer: queued in /hitl/queue
        Reviewer->>API: POST /hitl/review (corrections)
        API->>KG: phase3_maintenance (versioned update)
    end
```

The HITL queue is in-memory by default — wire it to your own store (Postgres, Redis, etc.) by mirroring `AppState.hitl_queue` writes.

---

## 📂 Project Layout

```
metadata-kg/
├── metadata_kg/
│   ├── core/
│   │   ├── kg_builder.py        # Knowledge Graph (rdflib + networkx)
│   │   ├── llm_agent.py         # LLM-as-Agent (LangChain + Claude)
│   │   └── metadata_schema.py   # DCAT 2 / DCMI Pydantic models
│   ├── pipeline/
│   │   ├── ingest.py            # text/JSON/YAML/PDF ingestion
│   │   ├── extract.py           # LLM-based metadata extraction
│   │   ├── validate.py          # KG consistency validation
│   │   └── lifecycle.py         # 4-phase lifecycle manager
│   ├── governance/
│   │   ├── lineage.py           # PROV-O lineage tracker
│   │   ├── policy.py            # Policy-as-code (GDPR, mandatory fields)
│   │   └── xai.py               # Explainability layer
│   ├── search/
│   │   └── semantic_search.py   # Sentence-Transformers + BM25 hybrid
│   ├── api/
│   │   └── routes.py            # FastAPI REST endpoints
│   └── tests/                   # pytest suite + benchmark.py
├── streamlit_app.py             # Web UI (Streamlit Cloud–ready)
├── examples/
│   ├── quickstart.py
│   ├── sample_dataset.json
│   └── custom_policy.yaml
├── docs/
│   ├── ARCHITECTURE.md
│   └── DEPLOYMENT.md
├── .streamlit/                  # Streamlit config + secrets template
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

### Environment variables (`.env`)

| Variable                 | Default                          | Purpose                                            |
|--------------------------|----------------------------------|----------------------------------------------------|
| `ANTHROPIC_API_KEY`      | _unset_                          | Enables LLM mode. Without it, deterministic agent. |
| `ANTHROPIC_MODEL`        | `claude-sonnet-4-20250514`       | Override the Claude model alias.                   |
| `API_HOST` / `API_PORT`  | `0.0.0.0` / `8000`               | FastAPI bind address.                              |
| `HF_HOME`                | `./.hf_cache`                    | Where Sentence-Transformers caches its model.      |
| `KG_STORAGE_DIR`         | `./data/kg`                      | Suggested Turtle dump directory.                   |
| `LINEAGE_STORAGE_DIR`    | `./data/lineage`                 | JSONL lineage persistence directory.               |

### Custom policy rules (`examples/custom_policy.yaml`)

```yaml
rules:
  - id: CUSTOM_LICENSE_REQUIRED
    description: "Dataset must declare a license."
    severity: error
    check: mandatory_field
    field: license

  - id: CUSTOM_NO_INTERNAL_URL
    description: "Public descriptions must not reference internal hostnames."
    severity: warning
    check: regex
    field: description
    pattern: "(internal|intranet|localhost|127\\.0\\.0\\.1)"
```

Load with `PolicyEngine().load_rules("examples/custom_policy.yaml")`.

---

## 📊 Benchmarks

Run the bundled benchmark:

```bash
python -m metadata_kg.tests.benchmark
# → benchmark_report.json
```

Reference results on the synthetic gold set bundled with the repo (4 extraction cases, 7 search queries):

| Metric             | Keyword (BM25) | Semantic | Hybrid |
|--------------------|---------------:|---------:|-------:|
| **Precision@5**    | 1.00           | 1.00     | 1.00   |
| **MRR@5**          | 1.00           | 1.00     | 1.00   |
| **Latency (ms)**   | 0.02           | 0.02     | 0.03   |
| **Title precision** (extraction) | — | — | 1.00 |
| **Key coverage** (extraction)    | — | — | 1.00 |

> ⚠️ The bundled gold set is intentionally small and illustrative. Replace `SEARCH_DOCS` and `SEARCH_QUERIES` in `benchmark.py` with your domain corpus for meaningful evaluation.

---

## 🧪 Testing

```bash
pytest -v                          # 46 tests in <1s
pytest --cov=metadata_kg           # with coverage
```

Test layout:

| File | Tests | Covers |
|------|------:|--------|
| `test_kg_builder.py`  | 8  | entity CRUD, validation, Turtle round-trip, cross-domain linking |
| `test_llm_agent.py`   | 9  | tools, agent run, HITL trigger, reasoning log |
| `test_lifecycle.py`   | 6  | all 4 phases + lineage emission |
| `test_governance.py`  | 12 | PII detection, policy YAML, lineage chain, XAI |
| `test_search.py`      | 6  | semantic/keyword/hybrid, query expansion, explain |
| `test_api.py`         | 7  | all FastAPI endpoints with `TestClient` |

---

## 📚 Standards & References

| Standard | Used in | Reference |
|----------|---------|-----------|
| **DCAT 2** | `core/metadata_schema.py` (entity types, mandatory fields) | <https://www.w3.org/TR/vocab-dcat-2/> |
| **DCMI Terms** | Schema namespace + 12-term Type vocabulary | <https://www.dublincore.org/specifications/dublin-core/dcmi-terms/> |
| **PROV-O** | `governance/lineage.py` (JSON-LD export) | <https://www.w3.org/TR/prov-o/> |
| **Pydantic v2** | All data models | <https://docs.pydantic.dev/> |

---

## 🗺️ Roadmap

- [ ] Persistent KG backend (SQLite / Postgres + Apache Jena)
- [ ] Real-time SPARQL endpoint
- [ ] GNN training pipeline using `kg.to_networkx()`
- [ ] Multi-tenant FastAPI mode
- [ ] PDF table extraction (current PDF reader is text-only)
- [ ] Active learning loop on HITL corrections
- [ ] Multilingual entity extraction (Thai / Lao / Khmer / Burmese)
- [ ] Docker Compose with FastAPI + Streamlit + Postgres

Contributions for any of these are very welcome — see [Contributing](#-contributing).

---

## 🤝 Contributing

1. Fork & clone.
2. `pip install -e ".[dev]"` then `pre-commit install` (optional once we add the hook).
3. Create a feature branch: `git checkout -b feature/<name>`.
4. Run the tests: `pytest -v`.
5. Format / lint: `ruff check . && ruff format .`.
6. Open a PR with a clear description. Reference any related issue.

For substantial features, please open an issue first to discuss scope.

---

## 🔒 Security

This project has access controls only at the application layer. **Do not** expose the FastAPI server to the public internet without an authentication layer (e.g. OAuth2, mTLS, an upstream API gateway).

Report security issues privately via email to <wirach@kku.ac.th>.

---

## 📜 Citation

If you use Metadata KG in academic work, please cite:

```bibtex
@software{chansanam2026metadata_kg,
  author       = {Chansanam, Wirapong},
  title        = {Metadata KG: AI Metadata Automation with Knowledge Graphs},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/<your-username>/metadata-kg},
  version      = {0.1.0},
  orcid        = {0000-0001-5546-8485}
}
```

---

## 📄 License

[MIT](LICENSE) © 2026 [Wirapong Chansanam](https://orcid.org/0000-0001-5546-8485), Khon Kaen University.

---

## 🙏 Acknowledgements

Built on the shoulders of giants — thank you to the maintainers of
[**rdflib**](https://rdflib.readthedocs.io) ·
[**networkx**](https://networkx.org) ·
[**LangChain**](https://www.langchain.com) ·
[**Anthropic Claude**](https://www.anthropic.com) ·
[**sentence-transformers**](https://www.sbert.net) ·
[**rank_bm25**](https://github.com/dorianbrown/rank_bm25) ·
[**FastAPI**](https://fastapi.tiangolo.com) ·
[**Streamlit**](https://streamlit.io) ·
[**Pydantic**](https://docs.pydantic.dev).

<div align="center">

---

**Made with 🐾 at Khon Kaen University**

⭐ Star this repo if you find it useful · 🐛 Open an issue · 🤝 PRs welcome

</div>
