import logging
from typing import Any, Dict, Generator, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from litellm import completion, RateLimitError, AuthenticationError, APIError, ModelResponse

logger = logging.getLogger(__name__)


class LLMClient:

    def __init__(self, provider: str, api_key: str, model: str):
        if not provider:
            raise ValueError("Provider must be specified.")
        if not api_key:
            raise ValueError("API key must be provided.")
        if not model:
            raise ValueError("Model must be specified.")

        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model

        logger.debug(f"LLMClient initialized for provider={self.provider}, model={self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=(
            retry_if_exception_type(RateLimitError) |
            retry_if_exception_type(APIError)
        )
    )
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        try:
            logger.debug(f"Sending chat request to {self.provider}/{self.model}")
            response: ModelResponse = completion(
                model=f"{self.provider}/{self.model}",
                messages=messages,
                api_key=self.api_key,
                **kwargs
            )
            content = response["choices"][0]["message"]["content"]
            logger.debug(f"Received response: {content[:120]!r}")
            return content

        except RateLimitError as e:
            logger.warning("Rate limit reached — retrying...")
            raise e
        except AuthenticationError:
            logger.error("Invalid API key or unauthorized access.")
            raise
        except APIError as e:
            logger.error(f"Provider API error: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during chat: {e}")
            raise

    def stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        try:
            logger.debug(f"Starting stream with {self.provider}/{self.model}")
            for chunk in completion(
                model=f"{self.provider}/{self.model}",
                messages=messages,
                api_key=self.api_key,
                stream=True,
                **kwargs
            ):
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
        except Exception as e:
            logger.exception(f"Error during streaming: {e}")
            raise

    def list_model(self) -> List[str]:
        
        return [self.model]


__all__ = ["LLMClient"]
