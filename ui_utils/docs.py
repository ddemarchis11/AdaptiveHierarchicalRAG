from typing import Any, Dict, List
import streamlit as st


def normalize_docs_for_ui(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for d in docs or []:
        if not isinstance(d, dict):
            continue
        out.append(
            {
                "id": d.get("id", d.get("chunk_id", "")),
                "text": d.get("text", ""),
                "metadata": d.get("metadata", {}) or {},
                "score": d.get("score", d.get("prior_rrf", d.get("metadata", {}).get("score_ppr"))),
            }
        )
    return out


def render_used_sources(docs: List[Dict[str, Any]]):
    docs = docs or []
    if not docs:
        st.caption("Nessuna sorgente recuperata.")
        return

    for d in docs:
        md = d.get("metadata", {}) or {}
        doc_id = md.get("doc_id", md.get("source", "unknown"))
        pos = md.get("position", md.get("chunk_rel_id", ""))
        score = d.get("score", None)

        header = f"**{doc_id}**"
        if pos != "":
            header += f" — pos {pos}"
        if score is not None:
            try:
                header += f" — score {float(score):.4f}"
            except Exception:
                pass

        st.markdown(header)
        st.markdown(d.get("text", ""))
        st.divider()
