import pytest
import logging
from turtle_cli.tools.executor import ToolExecutor
from turtle_cli.tools.protocol import ToolResult


class DummyTool:
    def __init__(self, should_succeed=True, raise_exception=False):
        self.should_succeed = should_succeed
        self.raise_exception = raise_exception
        self.timeout = 5
        self.executed_with = None

    def execute(self, **kwargs):
        self.executed_with = kwargs
        if self.raise_exception:
            raise ValueError("Simulated failure")
        if self.should_succeed:
            return ToolResult(True, data="Success")
        return ToolResult(False, error="Execution failed")


class DummyRegistry:
    def __init__(self, tools):
        self.tools = tools

    def get(self, name):
        return self.tools.get(name)


def test_execute_success():
    tool = DummyTool(should_succeed=True)
    registry = DummyRegistry({"my_tool": tool})
    executor = ToolExecutor(registry, timeout=10)

    result = executor.execute("my_tool", param="value")

    assert result.success
    assert result.data == "Success"
    assert tool.executed_with == {"param": "value"}
    assert tool.timeout == 5  # timeout restored


def test_execute_tool_not_found(caplog):
    registry = DummyRegistry({})
    executor = ToolExecutor(registry)

    with caplog.at_level(logging.ERROR):
        result = executor.execute("missing_tool")

    assert not result.success
    assert "not found in registry" in result.error
    assert "not found in registry" in caplog.text


def test_execute_failure():
    tool = DummyTool(should_succeed=False)
    registry = DummyRegistry({"fail_tool": tool})
    executor = ToolExecutor(registry)

    result = executor.execute("fail_tool")

    assert not result.success
    assert result.error == "Execution failed"
    assert tool.timeout == 5


def test_execute_with_exception(caplog):
    tool = DummyTool(raise_exception=True)
    registry = DummyRegistry({"crash_tool": tool})
    executor = ToolExecutor(registry)

    with caplog.at_level(logging.ERROR):
        result = executor.execute("crash_tool", arg=42)

    assert not result.success
    assert "Unexpected error executing tool" in result.error
    assert "Simulated failure" in result.error
    assert tool.timeout == 5


def test_execute_without_timeout():
    class NoTimeoutTool:
        def execute(self, **kwargs):
            return ToolResult(True, data="No timeout tool")

    registry = DummyRegistry({"simple_tool": NoTimeoutTool()})
    executor = ToolExecutor(registry)

    result = executor.execute("simple_tool", x=1)

    assert result.success
    assert result.data == "No timeout tool"
