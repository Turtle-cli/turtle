import json
import pytest
from types import SimpleNamespace
from turtle_cli.tools.parser import ToolCallParser, ParsedToolCall


class TestToolCallParser:
    def test_parse_valid_tool_call_from_dict_response(self):
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "test_function",
                            "arguments": json.dumps({"key": "value"})
                        }
                    }]
                }
            }]
        }

        result = ToolCallParser.parse_tool_calls(response)
        assert len(result) == 1
        assert isinstance(result[0], ParsedToolCall)
        assert result[0].id == "call_1"
        assert result[0].function_name == "test_function"
        assert result[0].arguments == {"key": "value"}

    def test_parse_empty_tool_calls(self):
        response = {"choices": [{"message": {}}]}
        result = ToolCallParser.parse_tool_calls(response)
        assert result == []

    def test_parse_tool_calls_with_direct_tool_calls_key(self):
        response = {
            "tool_calls": [{
                "id": "direct_call",
                "function": {"name": "direct_func", "arguments": {"x": 1}}
            }]
        }
        result = ToolCallParser.parse_tool_calls(response)
        assert len(result) == 1
        assert result[0].function_name == "direct_func"
        assert result[0].arguments == {"x": 1}

    def test_parse_tool_calls_from_object_with_tool_calls_attr(self):
        response = SimpleNamespace(tool_calls=[
            {"id": "obj_call", "function": {"name": "obj_func", "arguments": '{"a": 10}'}}
        ])
        result = ToolCallParser.parse_tool_calls(response)
        assert len(result) == 1
        assert result[0].id == "obj_call"
        assert result[0].function_name == "obj_func"
        assert result[0].arguments == {"a": 10}

    def test_extract_tool_calls_with_invalid_response(self):
        assert ToolCallParser._extract_tool_calls({}) is None
        assert ToolCallParser._extract_tool_calls(SimpleNamespace()) is None

    def test_parse_single_tool_call_with_invalid_json(self):
        tool_call = {
            "id": "bad_json",
            "function": {"name": "broken", "arguments": "{invalid json}"}
        }
        result = ToolCallParser._parse_single_tool_call(tool_call)
        assert result.id == "bad_json"
        assert result.function_name == "broken"
        assert result.arguments == {}

    def test_parse_single_tool_call_with_non_str_arguments(self):
        tool_call = {
            "id": "non_str",
            "function": {"name": "func", "arguments": {"k": "v"}}
        }
        result = ToolCallParser._parse_single_tool_call(tool_call)
        assert result.arguments == {"k": "v"}

    def test_parse_single_tool_call_with_missing_function(self):
        tool_call = {"id": "no_func"}
        result = ToolCallParser._parse_single_tool_call(tool_call)
        assert result.id == "no_func"
        assert result.function_name == ""
        assert result.arguments == {}

    def test_parse_single_tool_call_with_type_error(self):
        result = ToolCallParser._parse_single_tool_call(None)
        assert result is None

    def test_parse_tool_calls_with_empty_list(self):
        response = {"choices": [{"message": {"tool_calls": []}}]}
        assert ToolCallParser.parse_tool_calls(response) == []
