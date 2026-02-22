from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    SearchIndex
)

from .config import Settings
from .azure_project import AzureContext

def create_or_update_index(ctx: AzureContext, settings: Settings) -> str:
    index_client = SearchIndexClient(endpoint=ctx.search_endpoint, credential=ctx.search_key_credential)

    fields = [
        SearchField(name="parent_id", type=SearchFieldDataType.String),
        SearchField(name="title", type=SearchFieldDataType.String),

        SearchField(
            name="chunk_id",
            type=SearchFieldDataType.String,
            key=True,
            sortable=True,
            filterable=True,
            facetable=False,
            analyzer_name="keyword",
        ),
        SearchField(
            name="chunk",
            type=SearchFieldDataType.String,
            sortable=False,
            filterable=False,
            facetable=False,
            analyzer_name="en.lucene",
        ),
        SearchField(
            name="text_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=1536,
            vector_search_profile_name="HNSWprofile",
        ),
]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                parameters=HnswParameters(metric="cosine")
            ),
        ],
        profiles=[
            VectorSearchProfile(
                name="HNSWprofile",
                algorithm_configuration_name="myHnsw",
                vectorizer_name="ada-002",
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="ada-002",
                kind="azureOpenAI",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=settings.azure_openai_account,
                    deployment_name=settings.embedding_model_name,
                    api_key=ctx.openai_api_key,
                    model_name=settings.embedding_model_name,
                ),
            ),
        ],
    )

    index = SearchIndex(name=settings.index_name, fields=fields, vector_search=vector_search)
    result = index_client.create_or_update_index(index)
    return result.name
