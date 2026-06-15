"""Model-agnostic LLM factory. Swap providers via LLM_PROVIDER env var."""
import os
from langchain_core.language_models import BaseChatModel
from ca_agent.config import (
    LLM_PROVIDER, PLANNER_MODEL, SPECIALIST_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL
)


def get_llm(tier: str = "specialist") -> BaseChatModel:
    """
    Returns the appropriate LangChain chat model.

    tier="planner"    -> cheap, fast model (routing, drafting, classification)
    tier="specialist" -> smart model (complex reasoning, break analysis, M&A)

    temperature=0 everywhere — financial calculations must be deterministic.
    """
    provider = LLM_PROVIDER.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = SPECIALIST_MODEL if tier == "specialist" else PLANNER_MODEL
        return ChatOpenAI(model=model, temperature=0)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = "claude-sonnet-4-5" if tier == "specialist" else "claude-haiku-3-5"
        return ChatAnthropic(model=model, temperature=0)  # type: ignore

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = "gemini-2.5-pro" if tier == "specialist" else "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(model=model, temperature=0)  # type: ignore

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{provider}'. "
        f"Valid options: openai, anthropic, google, ollama"
    )
