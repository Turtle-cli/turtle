from typing import Any, Dict, List, Optional
from .protocol import ToolResult


class LiteLLMFormatter:
    """Formats tool execution results for LiteLLM (OpenAI format) conversation context"""

    @staticmethod
    def format_tool_response(
        tool_call_id: str,
        result: ToolResult,
        tool_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if result.success:
            return LiteLLMFormatter._format_success_response(tool_call_id, result, tool_name)
        else:
            return LiteLLMFormatter._format_error_response(tool_call_id, result, tool_name)

    @staticmethod
    def _format_success_response(
        tool_call_id: str,
        result: ToolResult,
        tool_name: Optional[str] = None
    ) -> Dict[str, Any]:
        content = LiteLLMFormatter._serialize_data(result.data)

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
            "name": tool_name
        }

    @staticmethod
    def _format_error_response(
        tool_call_id: str,
        result: ToolResult,
        tool_name: Optional[str] = None
    ) -> Dict[str, Any]:
        error_content = f"Error: {result.error}" if result.error else "Unknown error"

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": error_content,
            "name": tool_name
        }

    @staticmethod
    def _serialize_data(data: Any) -> str:
        if data is None:
            return ""
        elif isinstance(data, str):
            return data
        elif isinstance(data, (dict, list)):
            import json
            try:
                return json.dumps(data)
            except (TypeError, ValueError):
                return str(data)
        else:
            return str(data)

    @staticmethod
    def format_multiple_responses(
        responses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return [
            LiteLLMFormatter.format_tool_response(
                resp["tool_call_id"],
                resp["result"],
                resp.get("tool_name")
            )
            for resp in responses
        ]