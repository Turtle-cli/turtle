import os
import pytest
from unittest.mock import patch, MagicMock
from turtle_cli.llm.client import LLMClient


@pytest.fixture
def llm_client(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "fake_api_key")
    return LLMClient(provider="openai", api_key=os.getenv("LITELLM_API_KEY"), model="gpt-3.5-turbo")

def test_init_raises_when_provider_missing():
    with pytest.raises(ValueError, match="Provider must be specified."):
        LLMClient(provider="", api_key="key", model="gpt-3.5-turbo")


def test_init_raises_when_api_key_missing():
    with pytest.raises(ValueError, match="API key must be provided."):
        LLMClient(provider="openai", api_key="", model="gpt-3.5-turbo")


def test_init_raises_when_model_missing():
    with pytest.raises(ValueError, match="Model must be specified."):
        LLMClient(provider="openai", api_key="key", model="")

@patch("turtle_cli.llm.client.completion")
def test_chat_success(mock_completion, llm_client):
    mock_completion.return_value = {
        "choices": [{"message": {"content": "Hello, world!"}}]
    }

    response = llm_client.chat(messages=[{"role": "user", "content": "Hi!"}])

    assert response == "Hello, world!"
    mock_completion.assert_called_once_with(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hi!"}],
        api_key="fake_api_key"
    )

@patch("turtle_cli.llm.client.completion")
def test_chat_rate_limit_retry(mock_completion, llm_client):
    from litellm import RateLimitError

    mock_completion.side_effect = [
        RateLimitError("openai", "gpt-3.5-turbo", "Rate limit"),
        RateLimitError("openai", "gpt-3.5-turbo", "Rate limit"),
        {"choices": [{"message": {"content": "Success after retry"}}]},
    ]

    response = llm_client.chat(messages=[{"role": "user", "content": "Retry test"}])
    assert response == "Success after retry"
    assert mock_completion.call_count == 3


@patch("turtle_cli.llm.client.completion")
def test_chat_raises_auth_error(mock_completion, llm_client):
    from litellm import AuthenticationError

    mock_completion.side_effect = AuthenticationError(
        "openai", "gpt-3.5-turbo", "Invalid key"
    )

    with pytest.raises(AuthenticationError):
        llm_client.chat(messages=[{"role": "user", "content": "Hi!"}])

@patch("turtle_cli.llm.client.completion")
def test_chat_unexpected_error(mock_completion, llm_client):
    mock_completion.side_effect = Exception("Unexpected failure")

    with pytest.raises(Exception):
        llm_client.chat(messages=[{"role": "user", "content": "Hi!"}])


@patch("turtle_cli.llm.client.completion")
def test_stream_success(mock_completion, llm_client):
    mock_completion.return_value = iter([
        {"choices": [{"delta": {"content": "Hello "}}]},
        {"choices": [{"delta": {"content": "World"}}]}
    ])

    collected = list(llm_client.stream(messages=[{"role": "user", "content": "Hi!"}]))

    assert "".join(collected) == "Hello World"
    mock_completion.assert_called_once_with(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hi!"}],
        api_key="fake_api_key",
        stream=True
    )

@patch("turtle_cli.llm.client.completion")
def test_chat_raises_api_error(mock_completion, llm_client):
    from litellm import APIError
    from tenacity import RetryError

    mock_completion.side_effect = APIError(
        500, "API down", "openai", "gpt-3.5-turbo"
    )

    with pytest.raises(RetryError) as exc_info:
        llm_client.chat(messages=[{"role": "user", "content": "test"}])
    
    assert isinstance(exc_info.value.last_attempt.exception(), APIError)

def test_chat_empty_messages(llm_client):
    with pytest.raises(ValueError, match="Messages list cannot be empty."):
        llm_client.chat(messages=[])

def test_stream_empty_messages(llm_client):
    with pytest.raises(ValueError, match="Messages list cannot be empty."):
        list(llm_client.stream(messages=[]))

@patch("turtle_cli.llm.client.completion")
def test_stream_raises_exception(mock_completion, llm_client):
    mock_completion.side_effect = Exception("Stream failure")

    with pytest.raises(Exception, match="Stream failure"):
        list(llm_client.stream(messages=[{"role": "user", "content": "Hi!"}]))

def test_list_model(llm_client):
    assert llm_client.list_model() == ["gpt-3.5-turbo"]
