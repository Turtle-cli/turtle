from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Type, Union
import json


@dataclass
class ToolParameter:
    """Parameter definition for tool schema validation"""
    name: str
    type: Type
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolSchema:
    """Schema definition for a tool with parameter specifications"""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)

    def to_openai_format(self) -> Dict[str, Any]:
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": self._python_type_to_json(param.type),
                "description": param.description
            }
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def _python_type_to_json(self, py_type: Type) -> str:
        if py_type == str:
            return "string"
        elif py_type == int:
            return "integer"
        elif py_type == float:
            return "number"
        elif py_type == bool:
            return "boolean"
        elif py_type == list or py_type == List:
            return "array"
        elif py_type == dict or py_type == Dict:
            return "object"
        else:
            return "string"


@dataclass
class ToolResult:
    """Standardized result from tool execution"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Base protocol for all tools"""

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass


class ToolRegistry:
    """Registry for tool registration and lookup"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.schema.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_schemas(self) -> List[ToolSchema]:
        return [tool.schema for tool in self._tools.values()]

    def export_openai_format(self) -> List[Dict[str, Any]]:
        return [schema.to_openai_format() for schema in self.get_schemas()]