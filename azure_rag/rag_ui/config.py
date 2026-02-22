import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass(frozen=True)
class Settings:
    project_endpoint: str = os.getenv("PROJECT_ENDPOINT", "")
    openai_connection_name: str = os.getenv("OPENAI_CONNECTION_NAME", "")
    multiservice_connection_name: str = os.getenv("MULTISERVICE_CONNECTION_NAME", "")
    search_connection_name: str = os.getenv("SEARCH_CONNECTION_NAME", "")
    azure_openai_account: str = os.getenv("AZURE_OPENAI_ACCOUNT", "")
    connection_string_name: str = os.getenv("CONNECTION_STRING_NAME", "")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "")
    lm_name: str = os.getenv("LM_NAME", "")

    index_name: str = os.getenv("INDEX_NAME", "rag-tesi-index")
    blob_container_name: str = os.getenv("BLOB_CONTAINER_NAME", "all-documents")
    skillset_name: str = os.getenv("SKILLSET_NAME", "rag-tesi-skillset")
    indexer_name: str = os.getenv("INDEXER_NAME", "rag-tesi-idxr")

    openai_api_version: str = "2024-10-21"
    search_debug_api_version: str = "2025-05-01-preview"

def validate_settings(s: Settings) -> None:
    missing = []
    for k, v in s.__dict__.items():
        if k in {"index_name", "blob_container_name", "skillset_name", "indexer_name",
                 "openai_api_version", "search_debug_api_version"}:
            continue
        if not v:
            missing.append(k)
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
