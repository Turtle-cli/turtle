import pytest
from unittest.mock import MagicMock, patch
from src.turtle_cli.tools.adapters import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    ExecuteCommandTool
)
from src.turtle_cli.tools.protocol import ToolSchema, ToolResult


def test_read_file_tool_success(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello")

    tool = ReadFileTool(str(tmp_path))
    result = tool.execute(path="test.txt")

    assert result.success
    assert result.data == "Hello"
    assert result.error is None


def test_read_file_tool_missing_path():
    tool = ReadFileTool(".")
    result = tool.execute()
    assert not result.success
    assert "Path parameter is required" in result.error


def test_read_file_tool_file_not_found(tmp_path):
    tool = ReadFileTool(str(tmp_path))
    result = tool.execute(path="nonexistent.txt")
    assert not result.success
    assert "No such file" in result.error or "not found" in result.error


def test_read_file_tool_value_error(monkeypatch):
    tool = ReadFileTool(".")
    monkeypatch.setattr(tool.fs, "read_file", lambda _: (_ for _ in ()).throw(ValueError("Invalid path")))
    result = tool.execute(path="bad")
    assert not result.success
    assert "Invalid path" in result.error


def test_read_file_tool_unexpected_error(monkeypatch):
    tool = ReadFileTool(".")
    monkeypatch.setattr(tool.fs, "read_file", lambda _: (_ for _ in ()).throw(RuntimeError("Oops")))
    result = tool.execute(path="bad")
    assert not result.success
    assert "Unexpected error" in result.error


def test_write_file_tool_success(tmp_path):
    tool = WriteFileTool(str(tmp_path))
    result = tool.execute(path="output.txt", content="Hello")
    assert result.success
    assert "Successfully wrote" in result.data


def test_write_file_tool_missing_path():
    tool = WriteFileTool(".")
    result = tool.execute(content="Hello")
    assert not result.success
    assert "Path parameter is required" in result.error


def test_write_file_tool_missing_content():
    tool = WriteFileTool(".")
    result = tool.execute(path="file.txt")
    assert not result.success
    assert "Content parameter is required" in result.error


def test_write_file_tool_value_error(monkeypatch):
    tool = WriteFileTool(".")
    monkeypatch.setattr(tool.fs, "write_file", lambda p, c: (_ for _ in ()).throw(ValueError("Invalid write")))
    result = tool.execute(path="bad.txt", content="Hi")
    assert not result.success
    assert "Invalid write" in result.error


def test_write_file_tool_unexpected_error(monkeypatch):
    tool = WriteFileTool(".")
    monkeypatch.setattr(tool.fs, "write_file", lambda p, c: (_ for _ in ()).throw(RuntimeError("Crash")))
    result = tool.execute(path="bad.txt", content="Hi")
    assert not result.success
    assert "Unexpected error" in result.error


def test_list_directory_tool_success(tmp_path):
    (tmp_path / "a.txt").write_text("hi")
    tool = ListDirectoryTool(str(tmp_path))
    result = tool.execute()
    assert result.success
    filenames = [item["name"] for item in result.data]
    assert "a.txt" in filenames


def test_list_directory_tool_with_path(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("hi")
    tool = ListDirectoryTool(str(tmp_path))
    result = tool.execute(path="sub")
    assert result.success
    filenames = [item["name"] for item in result.data]
    assert "b.txt" in filenames


def test_list_directory_tool_file_not_found(tmp_path):
    tool = ListDirectoryTool(str(tmp_path))
    result = tool.execute(path="no_dir")
    assert not result.success
    assert "not found" in result.error or "No such file" in result.error


def test_list_directory_tool_value_error(monkeypatch):
    tool = ListDirectoryTool(".")
    monkeypatch.setattr(tool.fs, "list_directory", lambda _: (_ for _ in ()).throw(ValueError("Bad dir")))
    result = tool.execute(path="bad")
    assert not result.success
    assert "Bad dir" in result.error


def test_list_directory_tool_unexpected_error(monkeypatch):
    tool = ListDirectoryTool(".")
    monkeypatch.setattr(tool.fs, "list_directory", lambda _: (_ for _ in ()).throw(RuntimeError("Oops")))
    result = tool.execute(path="bad")
    assert not result.success
    assert "Unexpected error" in result.error


class MockCommandResult:
    def __init__(self, stdout="", stderr="", exit_code=0, timed_out=False):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.timed_out = timed_out


def test_execute_command_tool_success(monkeypatch):
    tool = ExecuteCommandTool(timeout=10)
    mock_result = MockCommandResult(stdout="ok", stderr="", exit_code=0)
    monkeypatch.setattr(tool.executor, "execute", lambda cmd: mock_result)

    result = tool.execute(command="echo hi")

    assert result.success
    assert result.data["stdout"] == "ok"
    assert result.error is None


def test_execute_command_tool_failure(monkeypatch):
    tool = ExecuteCommandTool()
    mock_result = MockCommandResult(stdout="", stderr="Error!", exit_code=1)
    monkeypatch.setattr(tool.executor, "execute", lambda cmd: mock_result)

    result = tool.execute(command="fail")

    assert not result.success
    assert result.error == "Error!"
    assert result.data["exit_code"] == 1


def test_execute_command_tool_missing_command():
    tool = ExecuteCommandTool()
    result = tool.execute()
    assert not result.success
    assert "Command parameter is required" in result.error


def test_execute_command_tool_unexpected_error(monkeypatch):
    tool = ExecuteCommandTool()
    monkeypatch.setattr(tool.executor, "execute", lambda _: (_ for _ in ()).throw(RuntimeError("Boom")))
    result = tool.execute(command="bad")
    assert not result.success
    assert "Unexpected error" in result.error


def test_read_file_tool_schema():
    tool = ReadFileTool()
    schema = tool.schema
    assert isinstance(schema, ToolSchema)
    assert schema.name == "read_file"
    assert "Read" in schema.description
    assert any(p.name == "path" for p in schema.parameters)


def test_write_file_tool_schema():
    tool = WriteFileTool()
    schema = tool.schema
    assert isinstance(schema, ToolSchema)
    assert schema.name == "write_file"
    assert "content" in schema.description
    assert {p.name for p in schema.parameters} == {"path", "content"}


def test_list_directory_tool_schema():
    tool = ListDirectoryTool()
    schema = tool.schema
    assert isinstance(schema, ToolSchema)
    assert schema.name == "list_directory"
    assert "directory" in schema.description
    param_names = [p.name for p in schema.parameters]
    assert "path" in param_names
    path_param = next(p for p in schema.parameters if p.name == "path")
    assert path_param.default == "."


def test_execute_command_tool_schema():
    tool = ExecuteCommandTool()
    schema = tool.schema
    assert isinstance(schema, ToolSchema)
    assert schema.name == "execute_command"
    assert "shell" in schema.description
    param_names = {p.name for p in schema.parameters}
    assert "command" in param_names
    assert "timeout" in param_names
