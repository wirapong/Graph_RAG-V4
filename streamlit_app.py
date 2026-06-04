"""Streamlit dashboard for Metadata KG.

Tabs:
1. Ingest — paste text → see extracted metadata + lineage
2. Search — semantic / keyword / hybrid search
3. Explain — XAI for a chosen entity
4. Governance — policy compliance + HITL queue
5. Graph — KG stats and Turtle export

Deployable to Streamlit Cloud out of the box (uses requirements.txt and .streamlit/).
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from metadata_kg.core.kg_builder import MetadataKnowledgeGraph
from metadata_kg.core.llm_agent import MetadataAgent
from metadata_kg.governance.lineage import DataLineage
from metadata_kg.governance.policy import PolicyEngine
from metadata_kg.governance.xai import ExplainabilityLayer
from metadata_kg.pipeline.lifecycle import MetadataLifecycle
from metadata_kg.search.semantic_search import SemanticMetadataSearch

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Metadata KG — DCAT 2 / DCMI",
    page_icon="🕸️",
    layout="wide",
)

# Pull secrets into env (so MetadataAgent picks them up)
if "ANTHROPIC_API_KEY" in st.secrets and not os.getenv("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
if "ANTHROPIC_MODEL" in st.secrets:
    os.environ["ANTHROPIC_MODEL"] = st.secrets["ANTHROPIC_MODEL"]


# ---------------------------------------------------------------------------
# Session state (one KG per session)
# ---------------------------------------------------------------------------
def _bootstrap_state() -> None:
    if "kg" not in st.session_state:
        st.session_state.kg = MetadataKnowledgeGraph(name="streamlit_kg")
        st.session_state.lineage = DataLineage()
        st.session_state.agent = MetadataAgent(kg=st.session_state.kg)
        st.session_state.lifecycle = MetadataLifecycle(
            kg=st.session_state.kg,
            agent=st.session_state.agent,
            lineage=st.session_state.lineage,
        )
        st.session_state.policy = PolicyEngine()
        st.session_state.xai = ExplainabilityLayer(st.session_state.kg, st.session_state.lineage)
        st.session_state.search = SemanticMetadataSearch()


_bootstrap_state()
kg = st.session_state.kg
lineage = st.session_state.lineage
lifecycle = st.session_state.lifecycle
policy = st.session_state.policy
xai = st.session_state.xai
search = st.session_state.search

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🕸️ Metadata KG — AI-Powered DCAT 2 Catalogue")
st.caption(
    "LLM-as-Agent extraction · KG-based validation · XAI traceability · HITL workflow"
)

with st.sidebar:
    st.markdown("### 📊 Live Stats")
    stats = kg.stats()
    st.metric("Triples", stats["triples"])
    st.metric("Entities", stats["entities"])
    st.metric("Lineage events", len(lineage))
    if st.button("🗑️ Reset KG", use_container_width=True):
        for key in ("kg", "lineage", "agent", "lifecycle", "policy", "xai", "search"):
            st.session_state.pop(key, None)
        st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_ingest, tab_search, tab_explain, tab_governance, tab_graph = st.tabs(
    ["📥 Ingest", "🔎 Search", "🧠 Explain (XAI)", "🛡️ Governance", "🗂️ Graph"]
)

# ============================ Ingest =================================
with tab_ingest:
    st.subheader("Ingest a document")
    sample = (
        "Title: Thailand Air Quality 2024\n"
        "Description: Hourly PM2.5 measurements from 80 stations across Thailand.\n"
        "Tags: air-quality, PM25, Thailand"
    )
    text = st.text_area("Paste raw text or structured metadata:", value=sample, height=160)
    source = st.text_input("Source label", value="streamlit-upload")

    if st.button("🚀 Run full pipeline (Phase 1 + Phase 2)"):
        with st.spinner("Extracting + validating..."):
            results = lifecycle.run_full(text, source=source)
            search.index_kg(kg)
        st.success(
            f"Created **{len(results['creation'].entity_ids)}** entities · "
            f"{len(results['creation'].events)} lineage events · "
            f"dups merged: {len(results['cleaning'].duplicates_merged)}"
        )
        st.json({
            "entity_ids": results["creation"].entity_ids,
            "events": results["creation"].events,
            "cleaning": {
                "duplicates_merged": results["cleaning"].duplicates_merged,
                "suggestions": results["cleaning"].suggestions,
            },
        })

# ============================ Search =================================
with tab_search:
    st.subheader("Semantic search")
    col_q, col_method, col_k = st.columns([6, 2, 1])
    with col_q:
        q = st.text_input("Query", value="air quality")
    with col_method:
        method = st.selectbox("Method", ("hybrid", "semantic", "keyword"))
    with col_k:
        top_k = st.number_input("top_k", 1, 50, 10)

    if st.button("🔎 Search"):
        if not kg.all_entities():
            st.warning("KG is empty. Ingest something first.")
        else:
            search.index_kg(kg)
            if method == "semantic":
                results = search.search(q, top_k=int(top_k))
            elif method == "keyword":
                results = search.keyword_fallback(q, top_k=int(top_k))
            else:
                results = search.hybrid_search(q, top_k=int(top_k))

            if not results:
                st.info("No results.")
            else:
                df = pd.DataFrame([r.to_dict() for r in results])
                st.dataframe(df, use_container_width=True)

            st.markdown("**Query expansions:**")
            st.code("\n".join(search.expand_query(q)[:6]))

# ============================ Explain =================================
with tab_explain:
    st.subheader("Why was this metadata produced?")
    if not kg.all_entities():
        st.warning("KG is empty. Ingest something first.")
    else:
        eid = st.selectbox("Choose an entity:", kg.all_entities())
        if eid:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(xai.explain_decision(eid))
            with col_b:
                st.markdown("**Confidence report**")
                st.json(xai.confidence_report(eid))

# ============================ Governance =============================
with tab_governance:
    st.subheader("Policy compliance")
    if kg.all_entities():
        eid = st.selectbox(
            "Entity to check",
            kg.all_entities(),
            key="gov_entity",
        )
        if eid:
            result = policy.check_compliance(kg.get_entity(eid))
            if result["pass"]:
                st.success("✅ Passes all policies.")
            else:
                st.error(f"❌ {result['summary']['error']} errors")
            st.json(result)

    st.divider()
    st.subheader("Manual policy check (ad-hoc metadata)")
    md_input = st.text_area(
        "JSON metadata",
        value=json.dumps({"id": "test", "title": "Demo", "description": "Open dataset"}, indent=2),
        height=120,
    )
    if st.button("Run policy check"):
        try:
            md = json.loads(md_input)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
        else:
            st.json(policy.check_compliance(md))

# ============================ Graph ==================================
with tab_graph:
    st.subheader("Knowledge Graph")
    st.json(kg.stats())

    if kg.all_entities():
        st.markdown("#### Turtle export")
        st.code(kg.export_to_turtle()[:5000], language="turtle")

        st.download_button(
            label="⬇️ Download .ttl",
            data=kg.export_to_turtle().encode("utf-8"),
            file_name="kg.ttl",
            mime="text/turtle",
        )

    st.markdown("#### Lineage (JSON-LD)")
    if len(lineage) > 0:
        doc = lineage.export_lineage_graph()
        st.code(json.dumps(doc, indent=2)[:5000])
        st.download_button(
            "⬇️ Download lineage.jsonld",
            data=json.dumps(doc, indent=2).encode("utf-8"),
            file_name="lineage.jsonld",
            mime="application/ld+json",
        )
    else:
        st.info("No lineage events yet.")
