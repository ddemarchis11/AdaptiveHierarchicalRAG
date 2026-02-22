from dataclasses import dataclass
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential

from .config import Settings

@dataclass(frozen=True)
class AzureContext:
    project_client: AIProjectClient
    openai_api_key: str
    multi_service_key: str
    storage_connection_string: str
    search_endpoint: str
    search_key_credential: AzureKeyCredential

def build_azure_context(settings: Settings) -> AzureContext:
    project_client = AIProjectClient(
        credential=DefaultAzureCredential(),
        endpoint=settings.project_endpoint
    )

    openai_api_key = project_client.connections.get(
        settings.openai_connection_name,
        include_credentials=True
    ).credentials["key"]

    multi_service_key = project_client.connections.get(
        settings.multiservice_connection_name,
        include_credentials=True
    ).credentials["key"]

    storage_connection_string = project_client.connections.get(
        settings.connection_string_name,
        include_credentials=True
    ).credentials["connection_string"]

    search_connection = project_client.connections.get(
        settings.search_connection_name,
        include_credentials=True
    )
    search_endpoint = search_connection.target
    search_key_credential = AzureKeyCredential(search_connection.credentials["key"])

    return AzureContext(
        project_client=project_client,
        openai_api_key=openai_api_key,
        multi_service_key=multi_service_key,
        storage_connection_string=storage_connection_string,
        search_endpoint=search_endpoint,
        search_key_credential=search_key_credential
    )
