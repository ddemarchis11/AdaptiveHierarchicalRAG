from typing import List, Dict, Any

SEPARATOR = "####"

def format_sources(docs):
    parts = []
    for d in docs:
        title = d.get("title", "")
        chunk = d.get("chunk", "")
        parts.append(f"TITLE: {title}, CONTENT: {chunk}")
    return f"\n{SEPARATOR}\n".join(parts)

def default_system_prompt(with_history: bool = True) -> str:
    if with_history:
        return (
            "You are an AI assistant that must answer ONLY using the provided SOURCES and HISTORY.\n"
            "If the answer is neither in the SOURCES or the HISTORY, say 'I don't know'.\n"
            "Use bullets if the answer has multiple points, and keep summaries concise if the answer is long."
        )
    return (
        "You are an AI assistant that must answer ONLY using the provided sources.\n"
        "If the answer is not in the sources, say 'I don't know'.\n"
        "Use bullets if the answer has multiple points, and keep summaries concise if the answer is long."
    )
