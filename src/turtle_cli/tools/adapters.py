from typing import Any, Optional
from .protocol import Tool, ToolSchema, ToolParameter, ToolResult
from .filesystem import FileSystem
from .command import CommandExecutor


class ReadFileTool(Tool):
    """Tool adapter for reading files using FileSystem"""

    def __init__(self, working_dir: str = "."):
        self.fs = FileSystem(working_dir)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_file",
            description="Read the contents of a file",
            parameters=[
                ToolParameter("path", str, "Path to the file to read")
            ]
        )

    def execute(self, **kwargs) -> ToolResult:
        try:
            path = kwargs.get("path")
            if not path:
                return ToolResult(False, error="Path parameter is required")

            content = self.fs.read_file(path)
            return ToolResult(True, data=content)

        except FileNotFoundError as e:
            return ToolResult(False, error=str(e))
        except ValueError as e:
            return ToolResult(False, error=str(e))
        except Exception as e:
            return ToolResult(False, error=f"Unexpected error: {str(e)}")


class WriteFileTool(Tool):
    """Tool adapter for writing files using FileSystem"""

    def __init__(self, working_dir: str = "."):
        self.fs = FileSystem(working_dir)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_file",
            description="Write content to a file",
            parameters=[
                ToolParameter("path", str, "Path to the file to write"),
                ToolParameter("content", str, "Content to write to the file")
            ]
        )

    def execute(self, **kwargs) -> ToolResult:
        try:
            path = kwargs.get("path")
            content = kwargs.get("content")

            if not path:
                return ToolResult(False, error="Path parameter is required")
            if content is None:
                return ToolResult(False, error="Content parameter is required")

            self.fs.write_file(path, content)
            return ToolResult(True, data=f"Successfully wrote to {path}")

        except ValueError as e:
            return ToolResult(False, error=str(e))
        except Exception as e:
            return ToolResult(False, error=f"Unexpected error: {str(e)}")


class ListDirectoryTool(Tool):
    """Tool adapter for listing directory contents using FileSystem"""

    def __init__(self, working_dir: str = "."):
        self.fs = FileSystem(working_dir)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_directory",
            description="List the contents of a directory",
            parameters=[
                ToolParameter("path", str, "Path to the directory to list", required=False, default=".")
            ]
        )

    def execute(self, **kwargs) -> ToolResult:
        try:
            path = kwargs.get("path", ".")

            items = self.fs.list_directory(path)
            return ToolResult(True, data=items)

        except FileNotFoundError as e:
            return ToolResult(False, error=str(e))
        except ValueError as e:
            return ToolResult(False, error=str(e))
        except Exception as e:
            return ToolResult(False, error=f"Unexpected error: {str(e)}")


class ExecuteCommandTool(Tool):
    """Tool adapter for executing shell commands using CommandExecutor"""

    def __init__(self, working_dir: Optional[str] = None, timeout: int = 30):
        self.executor = CommandExecutor(working_dir, timeout)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="execute_command",
            description="Execute a shell command",
            parameters=[
                ToolParameter("command", str, "Command to execute"),
                ToolParameter("timeout", int, "Timeout in seconds", required=False, default=30)
            ]
        )

    def execute(self, **kwargs) -> ToolResult:
        try:
            command = kwargs.get("command")
            timeout = kwargs.get("timeout", self.executor.timeout)

            if not command:
                return ToolResult(False, error="Command parameter is required")

            self.executor.timeout = timeout
            result = self.executor.execute(command)

            return ToolResult(
                success=result.exit_code == 0,
                data={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out
                },
                error=result.stderr if result.exit_code != 0 else None
            )

        except Exception as e:
            return ToolResult(False, error=f"Unexpected error: {str(e)}")
