# Adaptive and Hierarchical RAG

## Overview (thesis + repository)

This repository supports my Master’s thesis, focused on an experimental comparison of different **Retrieval Augmented Generation (RAG)** architectures, with particular attention to how **chunking** and **retrieval** impact the quality of the answers produced by an LLM.

In particular, the following approaches are compared:
- **Naive RAG**, “single-shot top-k” baseline
- **CoT RAG**, adaptive multi-step retrieval guided by a scratchpad
- **Graph-based / Hierarchical RAG**, graph structures and communities

> **Note on results:** metrics, datasets, analyses, and insights are discussed in the thesis report; this README is mainly focused on the produced **UI**.

---

## Streamlit UI

The UI is built with **Streamlit** and allows you to interact with the RAG architectures through:
- pipeline selection (**Naive / CoT / Graph**)
- choice of high-level parameters (e.g., **top-k**, **max steps** in CoT)
- visualization of the **sources used** (retrieved chunks)
- optional **debug** mode to inspect context and retriever behavior

In addition, the UI includes an **upload + text extraction** section: you can upload **.pdf** or **.txt** files, extract their text, and index it to make it queryable.

---

## Quickstart: run the UI

### 1) Prerequisites
- Python (recommended 3.10+)
- An **Elasticsearch** instance
- A **Neo4j** instance (required for Graph / GraphRAG mode)
- Python dependencies installed from `requirements.txt`

### 2) Install Python dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Start services (local)
To start **Elasticsearch** locally you can use the provided script:

```bash
./elastic-start-local/start.sh
```

> **Note**: the script requires Docker to be installed.

For **Neo4j**:

- **Neo4j installed locally**
  - install Neo4j (Desktop or server) and start it
  - verify that the **Bolt** service is enabled on port `7687`
  - create a local instance and a database `general` (unless you change the default config)
  - install the `Graph Data Science - GDS` plugin

> **Note**: this is the approach I used (full local) for my tests. The code is public and you can also consider using docker-compose at your discretion.

### 4) Configuration (config.py / config module)
The pipeline reads a `PipelineConfig` and uses:
- `cfg.es.index_name`
- `cfg.neo4j.uri`, `cfg.neo4j.user`, `cfg.neo4j.password`

If you want to change the default configuration, make sure that:
- `cfg.es.index_name` points to the correct index (default: `general`)
- `cfg.neo4j.uri` is consistent (typically `bolt://localhost:7687`)
- Neo4j username/password are the ones you chose (see the dedicated section below)

### 5) Start the UI
Run the main Streamlit file:

```bash
streamlit run streamlit_app.py
```

![UI Screenshot](img/ui_streamlit.png)

---

## UI usage guide (main focus)

### A) Settings Sidebar
In the sidebar you’ll find:
- **RAG Approach**: `Naive`, `CoT`, `Graph`
- **Final top-k**: how many chunks are shown/provided to generation as the final context
- **Max steps (CoT)**: visible only in CoT mode; controls the maximum steps of iterative retrieval
- **Active index**: the currently selected index (see the “general index” section)

### B) Upload and index documents (Load & Index)
With “Load & Index” you can:
1. upload one or more `.pdf` / `.txt` files
2. index them into the reference provided in the config by pressing **Index**

When you press **Index**:
- for PDFs, a **text extraction** is performed (e.g., via `pypdf`), while for TXT a direct read is done
- the extracted texts are converted into a JSONL batch and **indexed into Elasticsearch**
- the structures needed for GraphRAG are built in **Neo4j**
- the active index is set to `general` by default, unless the user directly changes it in `config.py`

### C) Chat: ask questions and read sources
- Enter a question in the chat.
- The system runs retrieval according to the selected mode and generates an answer.
- At the bottom, you’ll find the **📚 Used Sources** expander: it shows the chunks actually used as sources (transparency/grounding).
- **Restart** button: clears the conversation and starts over.

---

## Indices and naming: why everything revolves around `general`

### Elasticsearch: `general` index (default)
The pipeline, as mentioned, references an Elasticsearch index called **`general`** by default:
- ingestion from the UI indexes into `general`,
- the UI sets/displays the active index, which can still be changed

**Possible extension**: with small configuration changes it is possible to manage multiple distinct Elasticsearch indices, e.g., `finance`, `legal`, etc. That said, for the typical usage of an “average” user, keeping a single index is often the best choice: it simplifies the experience and makes upload/indexing more immediate, unless you introduce more complex logic such as topic classification.

### Neo4j: `general` database + credentials
For Graph mode, in addition to the ES index, you need a Neo4j database consistent with the GraphRAG pipeline.

In the current implementation, the construction of GraphRAG structures is conceptually associated with the “general” context (same naming as the ES case). In other words, **Elasticsearch `general`** + **Neo4j `general`** are the default combination.

**Possible extension:** create multiple Neo4j databases (one per corpus) or a single database with separate labels/namespaces per index.

---

## About the setup

No “one-click” scripts were included to completely simplify the setup. In general, there wasn’t a very simple way to guarantee full reusability regardless of the environment or execution needs; additionally, scripting the starting/stopping of services like Docker and Neo4j is often quite fragile, so it would have added excessive complexity without a real benefit. The goal of the thesis was not strictly to provide a tangible product, given its more experimental nature, but I still felt that the effort to provide a UI was worthwhile.

---

## Quick troubleshooting

- **Indexing in Graph Mode takes a long time**  
  Generally yes: it may take minutes, but it is perfectly normal and consistent with the structures that need to be created

- **You don’t see sources / empty index**  
  You first need to use “Load & Index” and index at least one document

- **PDFs with complex layout**  
  “Simple” text extraction can degrade the quality of the indexed text (headers/columns/tables mixed together)
