from langchain_groq import ChatGroq
from config import DEFAULT_CONFIG, PipelineConfig
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

cfg = DEFAULT_CONFIG.evaluation

def get_llm_client(config: PipelineConfig):
    llm_cfg = config.llm

    kwargs = {
        "groq_api_key": llm_cfg.api_key,
        "model_name": llm_cfg.model if llm_cfg.model else "openai/gpt-oss-20b",
        "temperature": llm_cfg.temperature,
    }

    if hasattr(llm_cfg, 'reasoning_effort') and llm_cfg.reasoning_effort is not None:
        kwargs["reasoning_effort"] = llm_cfg.reasoning_effort
            
    return ChatGroq(**kwargs)

def get_project_client() -> AIProjectClient:
    return AIProjectClient(
        credential=DefaultAzureCredential(),
        endpoint=cfg.project_endpoint,
    )

def get_eval_client():
    project_client = get_project_client()
    return project_client.get_openai_client(api_version=cfg.api_version)