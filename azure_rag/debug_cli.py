import re
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

from azure.core.exceptions import HttpResponseError
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents import SearchClient

from rag_ui.config import Settings, validate_settings
from rag_ui.azure_project import build_azure_context
from rag_ui.search_index import create_or_update_index
from rag_ui.indexer_resources import (
    create_or_update_datasource,
    create_or_update_skillset,
    create_or_update_indexer,
    run_indexer,
    get_indexer_last_status,
)
from rag_ui.retrieval import hybrid_search_sdk, hybrid_search_rest_debug
from rag_ui.llm import answer_with_rag


def _mask(s: str, keep: int = 6) -> str:
    if not s:
        return ""
    if len(s) <= keep * 2:
        return "*" * len(s)
    return s[:keep] + "*" * (len(s) - keep * 2) + s[-keep:]


def _extract_account_name_from_connstr(conn_str: str) -> str:
    m = re.search(r"AccountName=([^;]+)", conn_str or "")
    return m.group(1) if m else "(unknown)"


def _hdr(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def reset_resources(ixr_client: SearchIndexerClient, idx_client: SearchIndexClient, settings: Settings):
    _hdr("RESET")
    for fn, name in [
        (ixr_client.delete_indexer, settings.indexer_name),
        (ixr_client.delete_skillset, settings.skillset_name),
        (ixr_client.delete_data_source_connection, "indexer-storage-connection"),
        (idx_client.delete_index, settings.index_name),
    ]:
        try:
            fn(name)
            print("deleted:", name)
        except HttpResponseError as e:
            if getattr(e, "status_code", None) == 404:
                print("not found:", name)
            else:
                raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true")
    p.add_argument("--provision", action="store_true")
    p.add_argument("--run-indexer", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--query", type=str, default="what's NASA's website?")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--knn", type=int, default=5)
    p.add_argument("--rest-debug", action="store_true")
    p.add_argument("--save-json", type=str, default="")
    p.add_argument("--rag", action="store_true")
    args = p.parse_args()

    load_dotenv(override=True)

    _hdr("1) SETTINGS")
    settings = Settings()
    validate_settings(settings)
    print("INDEX_NAME:", settings.index_name)
    print("SKILLSET_NAME:", settings.skillset_name)
    print("INDEXER_NAME:", settings.indexer_name)
    print("BLOB_CONTAINER_NAME:", settings.blob_container_name)

    _hdr("2) CONTEXT")
    ctx = build_azure_context(settings)
    print("SEARCH_ENDPOINT:", ctx.search_endpoint)
    print("AZURE_OPENAI_ACCOUNT:", settings.azure_openai_account)
    print("EMBEDDING_DEPLOY:", settings.embedding_model_name)
    print("LM_NAME:", settings.lm_name)
    print("STORAGE_ACCOUNT:", _extract_account_name_from_connstr(ctx.storage_connection_string))
    print("STORAGE_CONNSTR(masked):", _mask(ctx.storage_connection_string))

    idx_client = SearchIndexClient(endpoint=ctx.search_endpoint, credential=ctx.search_key_credential)
    ixr_client = SearchIndexerClient(endpoint=ctx.search_endpoint, credential=ctx.search_key_credential)

    _hdr("3) EXISTING RESOURCES")
    try:
        idx = idx_client.get_index(settings.index_name)
        print("index exists:", settings.index_name, "fields:", len(idx.fields))
    except Exception as e:
        print("index missing:", settings.index_name, str(e))

    try:
        sk = ixr_client.get_skillset(settings.skillset_name)
        print("skillset exists:", settings.skillset_name, "skills:", len(sk.skills))
    except Exception as e:
        print("skillset missing:", settings.skillset_name, str(e))

    try:
        ds = ixr_client.get_data_source_connection("indexer-storage-connection")
        print("datasource exists:", "indexer-storage-connection")
        print("container:", getattr(ds.container, "name", None))
    except Exception as e:
        print("datasource missing:", str(e))

    try:
        ixr = ixr_client.get_indexer(settings.indexer_name)
        print("indexer exists:", settings.indexer_name)
        print("target_index_name:", ixr.target_index_name)
        print("skillset_name:", ixr.skillset_name)
        print("data_source_name:", ixr.data_source_name)
    except Exception as e:
        print("indexer missing:", settings.indexer_name, str(e))

    if args.reset:
        reset_resources(ixr_client, idx_client, settings)

    if args.provision:
        _hdr("4) PROVISION")
        ix_name = create_or_update_index(ctx, settings)
        print("index:", ix_name)
        ds_name = create_or_update_datasource(ctx, settings)
        print("datasource:", ds_name)
        sk_name = create_or_update_skillset(ctx, settings)
        print("skillset:", sk_name)
        ixr_name = create_or_update_indexer(ctx, settings, ds_name)
        print("indexer:", ixr_name)

    if args.run_indexer:
        _hdr("5) RUN INDEXER")
        run_indexer(ctx, settings)
        print("run requested")

    if args.status or args.run_indexer:
        _hdr("6) STATUS")
        st = get_indexer_last_status(ctx, settings)
        print(json.dumps(st, indent=2, ensure_ascii=False))

    _hdr("7) QUERY")
    try:
        from azure.search.documents import SearchClient
        sc = SearchClient(endpoint=ctx.search_endpoint, index_name=settings.index_name, credential=ctx.search_key_credential)
        probe = sc.search(search_text="*", top=1, include_total_count=True)
        print("total_count:", probe.get_count())

        docs = hybrid_search_sdk(
            ctx,
            settings,
            query=args.query,
            top_k_vec=args.knn,
            top=args.top,
            select_fields=["title", "chunk"],
        )
        print("query:", args.query)
        print("results:", len(docs))

        for i, d in enumerate(docs, start=1):
            print("-" * 60)
            print(i, d.get("title"), d.get("@search.score"))
            ch = (d.get("chunk") or "").replace("\n", " ")
            print(ch[:500])

        if args.rag:
            completion, _ = answer_with_rag(ctx, settings, args.query, docs, history=[])
            _hdr("7B) LLM")
            print(completion)

    except Exception as e:
        print("query failed:", str(e))

    _hdr("DONE")
    print(datetime.now().isoformat())


if __name__ == "__main__":
    main()
