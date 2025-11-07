import pytest
from typing import Any, Dict, List
from dataclasses import dataclass
from turtle_cli.tools.protocol import ToolParameter, ToolSchema, ToolResult, Tool, ToolRegistry


class DummyTool(Tool):
    @property
    def schema(self):
        return ToolSchema(
            name="dummy_tool",
            description="A dummy tool",
            parameters=[
                ToolParameter(name="x", type=int, description="value of x"),
                ToolParameter(name="y", type=str, description="value of y", required=False, default="test")
            ]
        )

    def execute(self, **kwargs) -> ToolResult:
        if "error" in kwargs:
            return ToolResult(success=False, error="error occurred")
        return ToolResult(success=True, data={"x": kwargs.get("x"), "y": kwargs.get("y", "test")})


class TestToolSystem:
    def test_toolparameter_creation(self):
        param = ToolParameter(name="a", type=str, description="desc", required=False, default="val")
        assert param.name == "a"
        assert param.default == "val"

    def test_tool_schema_to_openai_format(self):
        schema = ToolSchema(
            name="tool_name",
            description="A test tool",
            parameters=[
                ToolParameter(name="param1", type=str, description="desc1"),
                ToolParameter(name="param2", type=int, description="desc2", required=False)
            ]
        )
        result = schema.to_openai_format()
        assert result["function"]["name"] == "tool_name"
        assert "param1" in result["function"]["parameters"]["properties"]

    @pytest.mark.parametrize("py_type,json_type", [
        (str, "string"),
        (int, "integer"),
        (float, "number"),
        (bool, "boolean"),
        (list, "array"),
        (dict, "object"),
        (set, "string")
    ])
    def test_python_type_to_json(self, py_type, json_type):
        schema = ToolSchema(name="test", description="test")
        assert schema._python_type_to_json(py_type) == json_type

    def test_toolresult_success(self):
        result = ToolResult(success=True, data={"a": 1})
        assert result.success
        assert result.data == {"a": 1}

    def test_toolresult_failure(self):
        result = ToolResult(success=False, error="failed")
        assert not result.success
        assert result.error == "failed"

    def test_dummy_tool_execution_success(self):
        tool = DummyTool()
        res = tool.execute(x=5)
        assert res.success
        assert res.data["x"] == 5

    def test_dummy_tool_execution_failure(self):
        tool = DummyTool()
        res = tool.execute(error=True)
        assert not res.success
        assert res.error == "error occurred"

    def test_registry_register_and_get(self):
        registry = ToolRegistry()
        tool = DummyTool()
        registry.register(tool)
        assert registry.get("dummy_tool") == tool

    def test_registry_list_and_schemas(self):
        registry = ToolRegistry()
        tool = DummyTool()
        registry.register(tool)
        assert "dummy_tool" in registry.list_tools()
        schemas = registry.get_schemas()
        assert isinstance(schemas[0], ToolSchema)

    def test_registry_export_openai_format(self):
        registry = ToolRegistry()
        tool = DummyTool()
        registry.register(tool)
        exported = registry.export_openai_format()
        assert isinstance(exported, list)
        assert "function" in exported[0]
    
    def test_direct_call_abstract_methods(self):
        
        class Dummy(Tool):
            schema = None
            def execute(self, **kwargs):
                return ToolResult(success=True)

        assert Tool.schema.__get__(Dummy()) is None
        assert Tool.execute(Dummy(), test=1) is None
