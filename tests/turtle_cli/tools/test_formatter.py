import pytest
import json
from turtle_cli.tools.formatter import LiteLLMFormatter
from turtle_cli.tools.protocol import ToolResult


class TestLiteLLMFormatter:

    def test_format_success_response_with_string_data(self):
        result = ToolResult(success=True, data="Hello World")
        response = LiteLLMFormatter.format_tool_response("call_123", result, "test_tool")

        assert response["role"] == "tool"
        assert response["tool_call_id"] == "call_123"
        assert response["content"] == "Hello World"
        assert response["name"] == "test_tool"

    def test_format_success_response_with_dict_data(self):
        result = ToolResult(success=True, data={"status": "ok", "count": 42})
        response = LiteLLMFormatter.format_tool_response("call_456", result)

        expected_content = json.dumps({"status": "ok", "count": 42})
        assert response["role"] == "tool"
        assert response["tool_call_id"] == "call_456"
        assert response["content"] == expected_content
        assert response["name"] is None

    def test_format_success_response_with_list_data(self):
        result = ToolResult(success=True, data=[1, 2, 3])
        response = LiteLLMFormatter.format_tool_response("call_789", result)

        assert response["content"] == "[1, 2, 3]"

    def test_format_success_response_with_none_data(self):
        result = ToolResult(success=True, data=None)
        response = LiteLLMFormatter.format_tool_response("call_000", result)

        assert response["content"] == ""

    def test_format_error_response(self):
        result = ToolResult(success=False, error="File not found")
        response = LiteLLMFormatter.format_tool_response("call_err", result, "file_tool")

        assert response["role"] == "tool"
        assert response["tool_call_id"] == "call_err"
        assert response["content"] == "Error: File not found"
        assert response["name"] == "file_tool"

    def test_format_error_response_no_error_message(self):
        result = ToolResult(success=False, error=None)
        response = LiteLLMFormatter.format_tool_response("call_unknown", result)

        assert response["content"] == "Unknown error"

    def test_serialize_data_with_non_serializable_object(self):
        class CustomObject:
            def __str__(self):
                return "custom_object_str"

        result = ToolResult(success=True, data=CustomObject())
        response = LiteLLMFormatter.format_tool_response("call_obj", result)

        assert response["content"] == "custom_object_str"

    def test_format_multiple_responses(self):
        responses = [
            {
                "tool_call_id": "call_1",
                "result": ToolResult(success=True, data="result1"),
                "tool_name": "tool1"
            },
            {
                "tool_call_id": "call_2",
                "result": ToolResult(success=False, error="error2"),
                "tool_name": "tool2"
            }
        ]

        formatted = LiteLLMFormatter.format_multiple_responses(responses)

        assert len(formatted) == 2
        assert formatted[0]["content"] == "result1"
        assert formatted[0]["tool_call_id"] == "call_1"
        assert formatted[1]["content"] == "Error: error2"
        assert formatted[1]["tool_call_id"] == "call_2"

    def test_serialize_data_with_complex_dict_containing_non_serializable(self):
        class NonSerializable:
            pass

        data = {"key": NonSerializable()}
        content = LiteLLMFormatter._serialize_data(data)

        assert isinstance(content, str)
        assert "NonSerializable" in content or "object" in content
        