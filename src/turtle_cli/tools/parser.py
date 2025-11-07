import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class ParsedToolCall:
    """Parsed tool call with extracted function name and arguments"""
    id: str
    function_name: str
    arguments: Dict[str, Any]


class ToolCallParser:
    """Parses tool calls from LiteLLM unified response format"""

    @staticmethod
    def parse_tool_calls(response: Union[Dict[str, Any], Any]) -> List[ParsedToolCall]:
        
        tool_calls = ToolCallParser._extract_tool_calls(response)
        if not tool_calls:
            return []

        parsed_calls = []
        for tool_call in tool_calls:
            parsed_call = ToolCallParser._parse_single_tool_call(tool_call)
            if parsed_call:
                parsed_calls.append(parsed_call)

        return parsed_calls

    @staticmethod
    def _extract_tool_calls(response: Union[Dict[str, Any], Any]) -> Optional[List[Dict[str, Any]]]:
        """Extract tool_calls from response, handling both dict and message object"""
        if isinstance(response, dict):
            if "choices" in response and response["choices"]:
                message = response["choices"][0].get("message", {})
                return message.get("tool_calls")
            elif "tool_calls" in response:
                return response["tool_calls"]
        else:
            if hasattr(response, "tool_calls"):
                return response.tool_calls

        return None

    @staticmethod
    def _parse_single_tool_call(tool_call: Dict[str, Any]) -> Optional[ParsedToolCall]:
        """Parse a single tool call into ParsedToolCall"""
        try:
            call_id = tool_call.get("id", "")
            function_data = tool_call.get("function", {})
            function_name = function_data.get("name", "")
            arguments_str = function_data.get("arguments", "{}")

            try:
                arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
            except (json.JSONDecodeError, TypeError):
                arguments = {}

            return ParsedToolCall(
                id=call_id,
                function_name=function_name,
                arguments=arguments
            )
        except (KeyError, TypeError, AttributeError):
            return None
