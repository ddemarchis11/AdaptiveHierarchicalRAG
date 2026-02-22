import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Generator

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError("sentence-transformers must be installed.") from exc

from config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)

def load_model(model_name: str) -> SentenceTransformer:
    logger.info("Loading embedding model %s", model_name)
    return SentenceTransformer(model_name)

def iter_documents(path: Path) -> Iterable[tuple[str, str]]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = data.get("context") or data.get("text") or ""
            if not text: continue

            doc_id = data.get("corpus_name")
            if doc_id is None:
                doc_id = data.get("id") or f"doc_{i}"

            yield str(doc_id), text

def chunk_text(text: str, chunk_size_tokens: int, overlap_tokens: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tokens,
        chunk_overlap=overlap_tokens,
        encoding_name="cl100k_base",
    )
    return splitter.split_text(text)

def embed_texts(model: SentenceTransformer, texts: Sequence[str], batch_size: int) -> List[List[float]]:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()

def create_index_if_not_exists(client: Elasticsearch, index_name: str, embedding_dim: int):
    if client.indices.exists(index=index_name):
        logger.info(f"Index '{index_name}' already exists. Skipping creation.")
        return

    mapping = {
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id":   {"type": "keyword"},
                "position": {"type": "integer"},
                "text":     {"type": "text"},
                "vector": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    client.indices.create(index=index_name, body=mapping)
    logger.info(f"Created index '{index_name}' with vector dim {embedding_dim}")

def generate_actions(
    config: PipelineConfig, 
    model: SentenceTransformer
) -> Generator[dict, None, None]:
    chunk_size = config.chunking.chunk_size
    overlap = int(chunk_size * config.chunking.overlap_ratio)
    batch_size = config.embeddings.batch_size

    batch_texts = []
    batch_metas = []

    for doc_id, text in iter_documents(config.documents_path):
        chunks = chunk_text(text, chunk_size, overlap)
        
        for idx, chunk_text_val in enumerate(chunks):
            chunk_id = f"{doc_id}_c_{idx}"
            
            batch_texts.append(chunk_text_val)
            batch_metas.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "position": idx
            })

            if len(batch_texts) >= batch_size:
                embeddings = embed_texts(model, batch_texts, batch_size)
                
                for meta, txt, emb in zip(batch_metas, batch_texts, embeddings):
                    yield {
                        "_index": config.es.index_name,
                        "_id": meta["chunk_id"],
                        "_source": {
                            "chunk_id": meta["chunk_id"],
                            "doc_id": meta["doc_id"],
                            "position": meta["position"],
                            "text": txt,
                            "vector": emb
                        }
                    }
                
                batch_texts = []
                batch_metas = []

    if batch_texts:
        embeddings = embed_texts(model, batch_texts, batch_size)
        for meta, txt, emb in zip(batch_metas, batch_texts, embeddings):
            yield {
                "_index": config.es.index_name,
                "_id": meta["chunk_id"],
                "_source": {
                    "chunk_id": meta["chunk_id"],
                    "doc_id": meta["doc_id"],
                    "position": meta["position"],
                    "text": txt,
                    "vector": emb
                }
            }

def ingest_documents_es(config: PipelineConfig = DEFAULT_CONFIG):
    es_client = Elasticsearch(
        config.es.url,
        basic_auth=(config.es.user, config.es.password) if config.es.user else None,
        verify_certs=config.es.verify_certs
    )

    if not es_client.ping():
        raise ConnectionError(f"Cannot connect to Elasticsearch at {config.es.url}")

    model = load_model(config.embeddings.model_name)
    create_index_if_not_exists(es_client, config.es.index_name, config.embeddings.embedding_size)

    logger.info("Starting ingestion to Elasticsearch...")

    success_count = 0
    error_count = 0
    
    for ok, response in streaming_bulk(
        client=es_client,
        actions=generate_actions(config, model),
        chunk_size=500,
        max_retries=3
    ):
        if ok:
            success_count += 1
        else:
            error_count += 1
            logger.error("Error indexing doc: %s", response)

    logger.info(f"Ingestion complete. Indexed: {success_count}, Errors: {error_count}")
    es_client.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ingest_documents_es()