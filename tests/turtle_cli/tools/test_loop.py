import pytest
from unittest.mock import MagicMock, patch
from turtle_cli.tools.loop import ToolOrchestrator
from turtle_cli.tools.executor import ToolResult


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.chat.return_value = {
        "choices": [
            {"message": {"content": "Assistant reply"}}
        ]
    }
    return client


@pytest.fixture
def mock_conversation_manager():
    manager = MagicMock()
    manager.prepare_messages_for_api.return_value = [{"role": "user", "content": "hi"}]
    manager.get_conversation_summary.return_value = {"summary": "ok"}
    return manager


@pytest.fixture
def mock_tool_registry():
    registry = MagicMock()
    registry.export_openai_format.return_value = [{"name": "test_tool"}]
    return registry


@pytest.fixture
def orchestrator(mock_llm_client, mock_conversation_manager, mock_tool_registry):
    return ToolOrchestrator(
        llm_client=mock_llm_client,
        conversation_manager=mock_conversation_manager,
        tool_registry=mock_tool_registry,
        max_iterations=2
    )


def test_execute_loop_no_tool_calls(orchestrator, mock_llm_client):
    """Covers path with no tool calls found"""
    with patch("src.turtle_cli.tools.loop.ToolCallParser.parse_tool_calls", return_value=[]):
        result = orchestrator.execute_loop("Hello")
        assert result == "Assistant reply"
        assert orchestrator.iteration_count == 1


def test_extract_assistant_content_dict_case(orchestrator):
    """Covers dict response extraction"""
    response = {"choices": [{"message": {"content": "Hello"}}]}
    assert orchestrator._extract_assistant_content(response) == "Hello"


def test_extract_assistant_content_object_case(orchestrator):
    """Covers object-like response extraction"""
    class Dummy:
        def __init__(self):
            self.choices = [MagicMock()]
            self.choices[0].message.content = "Hi from object"

    response = Dummy()
    assert orchestrator._extract_assistant_content(response) == "Hi from object"


def test_extract_assistant_content_empty(orchestrator):
    """Covers return of empty string"""
    assert orchestrator._extract_assistant_content({"no_choices": []}) == ""


def test_reset_iteration_count(orchestrator):
    orchestrator.iteration_count = 5
    orchestrator.reset_iteration_count()
    assert orchestrator.iteration_count == 0


def test_get_conversation_state(orchestrator):
    """Ensures conversation state dictionary is correct"""
    state = orchestrator.get_conversation_state()
    assert set(state.keys()) == {"iteration_count", "max_iterations", "conversation_summary"}


def test_extract_assistant_content_invalid_type(orchestrator):
    """Covers case with unrelated type"""
    assert orchestrator._extract_assistant_content(None) == ""


def test_execute_tool_calls_direct(orchestrator):
    """Covers _execute_tool_calls path with assistant content and multiple tools"""
    parsed_tool_call = MagicMock()
    parsed_tool_call.id = "abc"
    parsed_tool_call.function_name = "fn"
    parsed_tool_call.arguments = {"x": 1}

    orchestrator.tool_executor.execute = MagicMock(return_value=ToolResult(success=True, data="done"))

    with patch("turtle_cli.tools.loop.LiteLLMFormatter.format_tool_response",
               return_value={"content": "formatted"}):
        orchestrator._execute_tool_calls([parsed_tool_call], {"choices": [{"message": {"content": "assistant msg"}}]})

    orchestrator.conversation_manager.add_message.assert_any_call("assistant", "assistant msg")
    orchestrator.conversation_manager.add_message.assert_any_call("tool", "formatted")


def test_execute_loop_max_iterations(orchestrator):
    """Covers max iteration limit reached"""
    orchestrator.max_iterations = 1
    parsed_tool_call = MagicMock()
    parsed_tool_call.function_name = "f"
    parsed_tool_call.arguments = {}
    parsed_tool_call.id = "id1"

    with patch("turtle_cli.tools.loop.ToolCallParser.parse_tool_calls", return_value=[parsed_tool_call]), \
         patch.object(orchestrator.tool_executor, "execute", return_value=ToolResult(success=True)), \
         patch("turtle_cli.tools.loop.LiteLLMFormatter.format_tool_response", return_value={"content": "tool output"}):
        result = orchestrator.execute_loop("Run")
        assert result == "Maximum iteration limit reached"
