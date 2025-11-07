import logging
import requests
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class ModelFetcher:

    def __init__(self):
        self.litellm_models_url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
        self.github_models_url = "https://api.github.com/repos/BerriAI/litellm/contents/litellm/llms"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _fetch_litellm_models(self) -> Optional[Dict[str, Any]]:
        try:
            logger.debug("Fetching models from LiteLLM model list")
            response = requests.get(self.litellm_models_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch LiteLLM models: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching models: {e}")
            return None

    def _extract_provider_models(self, models_data: Dict[str, Any], provider_name: str) -> List[str]:
        if not models_data:
            return []

        provider_models = []
        provider_key = provider_name.lower()

        for model_name, model_info in models_data.items():
            if isinstance(model_name, str):
                if model_name.startswith(f"{provider_key}/"):
                    model_id = model_name.replace(f"{provider_key}/", "")
                    provider_models.append(model_id)
                elif provider_key in model_name.lower():
                    provider_models.append(model_name)

        return sorted(list(set(provider_models)))

    def _get_fallback_models(self, provider_name: str) -> List[str]:
        fallback_models = {
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o"],
            "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            "gemini": ["gemini-pro", "gemini-pro-vision"],
            "vertex_ai": ["gemini-pro", "palm-2"],
            "azure": ["gpt-4", "gpt-35-turbo"],
            "cohere": ["command", "command-r", "embed-english-v2.0"],
            "huggingface": ["mistralai/Mistral-7B-Instruct-v0.1", "meta-llama/Llama-2-7b-chat-hf"],
            "groq": ["llama2-70b-4096", "mixtral-8x7b-32768"],
            "ollama": ["llama2", "mistral", "codellama"],
            "mistral": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
            "perplexity": ["pplx-7b-online", "pplx-70b-online"],
            "fireworks": ["accounts/fireworks/models/llama-v2-70b-chat", "accounts/fireworks/models/mixtral-8x7b-instruct"],
            "together": ["togethercomputer/llama-2-70b-chat", "togethercomputer/falcon-40b-instruct"],
            "replicate": ["meta/llama-2-70b-chat", "stability-ai/stable-diffusion"],
            "anyscale": ["meta-llama/Llama-2-70b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.1"],
            "deepinfra": ["meta-llama/Llama-2-70b-chat-hf", "codellama/CodeLlama-34b-Instruct-hf"],
            "palm": ["palm-2", "palm-2-chat"],
            "ai21": ["j2-ultra", "j2-mid", "j2-light"],
            "nlpcloud": ["chatdolphin", "finetuned-llama-2-70b"],
            "aleph_alpha": ["luminous-supreme", "luminous-extended"]
        }
        return fallback_models.get(provider_name.lower(), [])

    def get_models_for_provider(self, provider_name: str) -> List[str]:
        if not provider_name:
            logger.warning("Provider name is empty")
            return []

        logger.debug(f"Fetching models for provider: {provider_name}")

        models_data = self._fetch_litellm_models()
        if models_data:
            provider_models = self._extract_provider_models(models_data, provider_name)
            if provider_models:
                logger.debug(f"Found {len(provider_models)} models for {provider_name}")
                return provider_models

        logger.warning(f"Failed to fetch models from API, using fallback for {provider_name}")
        fallback_models = self._get_fallback_models(provider_name)

        if not fallback_models:
            logger.warning(f"No fallback models available for provider: {provider_name}")

        return fallback_models


def get_models_for_provider(provider_name: str) -> List[str]:
    fetcher = ModelFetcher()
    return fetcher.get_models_for_provider(provider_name)