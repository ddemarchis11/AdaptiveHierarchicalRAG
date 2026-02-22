from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SplitSkill,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    AzureOpenAIEmbeddingSkill,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    IndexProjectionMode,
    SearchIndexerSkillset,
    CognitiveServicesAccountKey,
    SearchIndexer,
)

from .azure_project import AzureContext
from .config import Settings

def create_or_update_datasource(ctx: AzureContext, settings: Settings) -> str:
    indexer_client = SearchIndexerClient(endpoint=ctx.search_endpoint, credential=ctx.search_key_credential)

    container = SearchIndexerDataContainer(name=settings.blob_container_name)

    data_source_connection = SearchIndexerDataSourceConnection(
        name="indexer-storage-connection",
        type="azureblob",
        connection_string=ctx.storage_connection_string,
        container=container,
    )
    ds = indexer_client.create_or_update_data_source_connection(data_source_connection)
    return ds.name

from azure.search.documents.indexes.models import (
    SplitSkill,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    AzureOpenAIEmbeddingSkill,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    IndexProjectionMode,
    SearchIndexerSkillset,
    CognitiveServicesAccountKey,
)

def create_or_update_skillset(ctx: AzureContext, settings: Settings) -> str:
    client = SearchIndexerClient(endpoint=ctx.search_endpoint, credential=ctx.search_key_credential)

    split_skill = SplitSkill(
        description="Split documents into chunks",
        text_split_mode="pages",
        context="/document",
        maximum_page_length=2000,
        page_overlap_length=200,
        inputs=[InputFieldMappingEntry(name="text", source="/document/content")],
        outputs=[OutputFieldMappingEntry(name="textItems", target_name="pages")],
    )

    embedding_skill = AzureOpenAIEmbeddingSkill(
        description="Generate embeddings via Azure OpenAI",
        context="/document/pages/*",
        resource_url=settings.azure_openai_account,
        deployment_name=settings.embedding_model_name,
        api_key=ctx.openai_api_key,
        model_name=settings.embedding_model_name,
        inputs=[InputFieldMappingEntry(name="text", source="/document/pages/*")],
        outputs=[OutputFieldMappingEntry(name="embedding", target_name="text_vector")],
    )

    index_projections = SearchIndexerIndexProjection(
        selectors=[
            SearchIndexerIndexProjectionSelector(
                target_index_name=settings.index_name,
                parent_key_field_name="parent_id",
                source_context="/document/pages/*",
                mappings=[
                    InputFieldMappingEntry(name="chunk", source="/document/pages/*"),
                    InputFieldMappingEntry(name="text_vector", source="/document/pages/*/text_vector"),
                    InputFieldMappingEntry(name="title", source="/document/metadata_storage_name"),
                ],
            ),
        ],
        parameters=SearchIndexerIndexProjectionsParameters(
            projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
        ),
    )

    cognitive_services_account = CognitiveServicesAccountKey(key=ctx.multi_service_key)

    skillset = SearchIndexerSkillset(
        name=settings.skillset_name,
        description="Minimal skillset: split + embedding",
        skills=[split_skill, embedding_skill],
        index_projection=index_projections,
        cognitive_services_account=cognitive_services_account,
    )

    client.create_or_update_skillset(skillset)
    return skillset.name

def create_or_update_indexer(ctx: AzureContext, settings: Settings, datasource_name: str) -> str:
    indexer_client = SearchIndexerClient(endpoint=ctx.search_endpoint, credential=ctx.search_key_credential)

    indexer = SearchIndexer(
        name=settings.indexer_name,
        description="Indexer to index documents and generate embeddings",
        skillset_name=settings.skillset_name,
        target_index_name=settings.index_name,
        data_source_name=datasource_name,
        parameters=None,
    )

    indexer_client.create_or_update_indexer(indexer)
    return indexer.name

def run_indexer(ctx: AzureContext, settings: Settings) -> None:
    client = SearchIndexerClient(endpoint=ctx.search_endpoint, credential=ctx.search_key_credential)
    client.run_indexer(settings.indexer_name)

def get_indexer_last_status(ctx: AzureContext, settings: Settings) -> dict:
    client = SearchIndexerClient(endpoint=ctx.search_endpoint, credential=ctx.search_key_credential)
    st = client.get_indexer_status(settings.indexer_name)
    last = getattr(st, "last_result", None)

    if last is None:
        return {
            "status": None,
            "error_message": None,
            "errors": [],
            "warnings": [],
            "start_time": None,
            "end_time": None,
            "items_processed": None,
            "items_failed": None,
        }

    return {
        "status": getattr(last, "status", None),
        "error_message": getattr(last, "error_message", None),
        "errors": [e.name for e in (getattr(last, "errors", None) or [])],
        "warnings": [w.message for w in (getattr(last, "warnings", None) or [])],
        "start_time": str(getattr(last, "start_time", "")),
        "end_time": str(getattr(last, "end_time", "")),
        "items_processed": getattr(last, "items_processed", None),
        "items_failed": getattr(last, "items_failed", None),
    }