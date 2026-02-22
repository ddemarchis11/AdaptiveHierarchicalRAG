from __future__ import annotations

import sys
import copy
import uuid
import inspect
import time
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable, Tuple

import streamlit as st

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from config import PipelineConfig
import index_es as index_module

DATA_DIR = Path("ui_data")
UPLOADS_DIR = DATA_DIR / "uploads"
EXTRACTED_DIR = DATA_DIR / "extracted"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

DEBUG_MODE = st.query_params.get("debug", "false").lower() == "true"

@dataclass
class ExtractedDoc:
    doc_id: str
    text: str


def _extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def _extract_document(saved_path: Path, doc_id: str) -> ExtractedDoc:
    suffix = saved_path.suffix.lower()
    if suffix == ".pdf":
        text = _extract_pdf(saved_path)
    elif suffix == ".txt":
        text = _extract_txt(saved_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return ExtractedDoc(doc_id=doc_id, text=text)


def _write_jsonl_for_indexer(docs: List[ExtractedDoc], out_path: Path) -> Path:
    import json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            rec = {"corpus_name": d.doc_id, "text": d.text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path


def _ingest_jsonl_into_general(cfg: PipelineConfig, jsonl_path: Path) -> Optional[str]:
    try:
        cfg2 = copy.deepcopy(cfg)
        cfg2.documents_path = jsonl_path
        cfg2.es.index_name = "general"
        index_module.ingest_documents_es(cfg2)
        return None
    except Exception:
        return traceback.format_exc()


def _call_with_optional_doc_ids(fn, *, cfg2: PipelineConfig, doc_ids: List[str]):
    sig = None
    try:
        sig = inspect.signature(fn)
    except Exception:
        sig = None

    try:
        if sig and "doc_ids" in sig.parameters:
            return fn(cfg2, doc_ids=doc_ids)
        return fn(cfg2)
    except TypeError:
        return fn(cfg2)


def _timeit_step(
    label: str,
    fn: Callable[[], Any],
    *,
    status_box=None,
) -> Tuple[Optional[Any], float, Optional[str]]:
    t0 = time.perf_counter()
    try:
        if status_box is not None:
            status_box.update(label=f"{label}...", state="running")

        res = fn()
        dt = time.perf_counter() - t0

        if status_box is not None:
            status_box.update(label=f"{label} ({dt:.2f}s)", state="complete")

        if DEBUG_MODE:
            st.write(f"**{label}** in {dt:.2f}s")

        return res, dt, None

    except Exception:
        dt = time.perf_counter() - t0
        tb = traceback.format_exc()

        if status_box is not None:
            status_box.update(label=f"{label} ({dt:.2f}s)", state="error")

        if DEBUG_MODE:
            st.error(f"{label} failed after {dt:.2f}s")
            st.code(tb)

        return None, dt, tb

def _build_graphrag_artifacts(
    cfg: PipelineConfig,
    jsonl_path: Path,
    doc_ids: List[str],
    *,
    status_box=None,
) -> Optional[str]:
    try:
        from graph_rag_utils.upsert_chunks_neo import ingest_chunks_neo4j_only
        from graph_rag_utils.chunk_graph_build import build_graph as build_chunk_graph
        from graph_rag_utils.community import compute_communities_and_summaries
        from graph_rag_utils.community_graph_build import build_community_graph, BuildParams

        cfg2 = copy.deepcopy(cfg)
        cfg2.documents_path = jsonl_path
        cfg2.es.index_name = "general"

        if DEBUG_MODE:
            st.write("### GraphRAG Debug")
            st.write("**doc_ids**:", doc_ids)
            st.write("**documents_path**:", str(cfg2.documents_path))
            st.write("**ES index**:", cfg2.es.index_name)
            st.write("**Neo4j DB**:", "general")

        _, _, err = _timeit_step(
            "Neo4j upsert chunks",
            lambda: _call_with_optional_doc_ids(ingest_chunks_neo4j_only, cfg2=cfg2, doc_ids=doc_ids),
            status_box=status_box,
        )
        if err:
            return err

        _, _, err = _timeit_step(
            "Build chunk graph edges (NEXT + SIMILAR_TO)",
            lambda: _call_with_optional_doc_ids(build_chunk_graph, cfg2=cfg2, doc_ids=doc_ids),
            status_box=status_box,
        )
        if err:
            return err

        _, _, err = _timeit_step(
            "Compute communities + summaries",
            lambda: _call_with_optional_doc_ids(compute_communities_and_summaries, cfg2=cfg2, doc_ids=doc_ids),
            status_box=status_box,
        )
        if err:
            return err

        params = BuildParams(
            top_k_neighbors=cfg2.community_similarity.top_k_neighbors,
            similarity_threshold=cfg2.community_similarity.similarity_threshold,
            cross_doc_only=cfg2.community_similarity.cross_doc_only,
            reset_edges=False,
        )

        def _run_comm_graph():
            sig = inspect.signature(build_community_graph)
            if "doc_ids" in sig.parameters:
                return build_community_graph(cfg2, params, doc_ids=doc_ids)
            return build_community_graph(cfg2, params)

        _, _, err = _timeit_step(
            "Build community similarity edges (COMM_SIMILAR_TO)",
            _run_comm_graph,
            status_box=status_box,
        )
        if err:
            return err

        return None

    except Exception:
        return traceback.format_exc()


def render_upload_and_index_box(cfg: PipelineConfig):
    st.subheader("Load & Index")
    uploads = st.file_uploader(
        "Load PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Index", use_container_width=True, disabled=not uploads):
        extracted_docs: List[ExtractedDoc] = []
        for uf in uploads or []:
            original_name = Path(uf.name)
            ext = original_name.suffix.lower()
            doc_id = f"{original_name.stem}-{uuid.uuid4().hex[:8]}"
            safe_name = f"{doc_id}{ext}"
            saved_path = UPLOADS_DIR / safe_name
            saved_path.write_bytes(uf.getbuffer())
            doc = _extract_document(saved_path, doc_id=doc_id)
            if doc.text.strip():
                extracted_docs.append(doc)

        if not extracted_docs:
            st.error("No extraction from uploaded files")
            return

        batch_path = EXTRACTED_DIR / f"general-batch-{uuid.uuid4().hex[:8]}.jsonl"
        _write_jsonl_for_indexer(extracted_docs, batch_path)
        batch_doc_ids = [d.doc_id for d in extracted_docs]
        
        with st.status("Indexing to Elasticsearch...", expanded=False) as status:
            t0 = time.perf_counter()
            err = _ingest_jsonl_into_general(cfg, batch_path)
            dt = time.perf_counter() - t0

            if err is not None:
                status.update(label=f"Indexing failed (ES) ({dt:.2f}s)", state="error")
                if DEBUG_MODE:
                    st.code(err)
                return

            status.update(label=f"Elasticsearch indexing completed ({dt:.2f}s)", state="complete")
            st.caption(f"Batch Indexed (ES): {batch_path}")

        with st.status("Building GraphRAG...", expanded=False) as status2:
            t0 = time.perf_counter()
            err2 = _build_graphrag_artifacts(cfg, batch_path, batch_doc_ids, status_box=status2)
            dt = time.perf_counter() - t0

            if err2 is not None:
                status2.update(label=f"GraphRAG build failed ({dt:.2f}s)", state="error")
                if DEBUG_MODE:
                    st.code(err2)
                return

            status2.update(label=f"GraphRAG build completed ({dt:.2f}s)", state="complete")
    
        st.session_state["general_index_ready"] = True
        st.session_state["active_index_name"] = "general"
        st.caption("Active index: general")