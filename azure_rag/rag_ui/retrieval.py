import json
import requests
from typing import List, Dict, Any, Optional

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from .azure_project import AzureContext
from .config import Settings

def hybrid_search_sdk(
    ctx: AzureContext,
    settings: Settings,
    query: str,
    top_k_vec: int = 5,
    top: int = 3,
    select_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if select_fields is None:
        select_fields = ["title", "chunk"]

    search_client = SearchClient(
        endpoint=ctx.search_endpoint,
        index_name=settings.index_name,
        credential=ctx.search_key_credential,
    )

    vector_query = VectorizableTextQuery(
        text=query,
        k_nearest_neighbors=top_k_vec,
        fields="text_vector"
    )

    results = search_client.search(
        search_text=query,
        search_mode="all",
        vector_queries=[vector_query],
        select=select_fields,
        top=top
    )

    out = []
    for r in results:
        item = {k: r.get(k) for k in select_fields}
        item["@search.score"] = r.get("@search.score")
        out.append(item)
    return out

def hybrid_search_rest_debug(
    ctx: AzureContext,
    settings: Settings,
    query: str,
    top_k_vec: int = 5,
    top: int = 3,
    save_json_path: Optional[str] = None
) -> Dict[str, Any]:
    url = f"{ctx.search_endpoint}/indexes/{settings.index_name}/docs/search?api-version={settings.search_debug_api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": ctx.search_key_credential.key,
    }
    body = {
        "search": query,
        "searchMode": "all",
        "searchFields": "chunk",
        "vectorQueries": [
            {
                "kind": "text",
                "text": query,
                "fields": "text_vector",
                "k": top_k_vec,
            }
        ],
        "top": top,
        "select": "chunk, title",
        "debug": "all",
    }

    resp = requests.post(url, headers=headers, data=json.dumps(body))
    resp.raise_for_status()
    data = resp.json()

    if save_json_path:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return data
