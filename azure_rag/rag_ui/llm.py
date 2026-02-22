from typing import List, Dict, Any, Tuple
from .azure_project import AzureContext
from .config import Settings
from .utils import format_sources, default_system_prompt

def get_openai_client(ctx: AzureContext, settings: Settings):
    return ctx.project_client.get_openai_client(api_version=settings.openai_api_version)

def answer_with_rag(
    ctx: AzureContext,
    settings: Settings,
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    history: List[Dict[str, str]] | None = None,
) -> Tuple[str, List[Dict[str, str]]]:

    if history is None:
        history = []

    sources_formatted = format_sources(retrieved_docs)
    system_prompt = default_system_prompt(with_history=True)

    openai_client = get_openai_client(ctx, settings)

    user_payload = f"QUERY: {query}\nSOURCES: {sources_formatted}\nHISTORY: {history}"

    resp = openai_client.chat.completions.create(
        model=settings.lm_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
    )

    completion = resp.choices[0].message.content

    updated = history + [
        {"role": "user", "content": f"QUERY: {query}\nSOURCES: {sources_formatted}"},
        {"role": "assistant", "content": completion},
    ]
    return completion, updated
