import logging
from typing import Any, Dict, Optional
from .protocol import Tool, ToolRegistry, ToolResult


class ToolExecutor:
    """Executes tools with error handling, timeout support, and logging"""

    def __init__(self, registry: ToolRegistry, timeout: int = 30):
        self.registry = registry
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        self.logger.info(f"Executing tool: {tool_name}")

        tool = self.registry.get(tool_name)
        if not tool:
            error_msg = f"Tool '{tool_name}' not found in registry"
            self.logger.error(error_msg)
            return ToolResult(False, error=error_msg)

        try:
            if hasattr(tool, 'timeout'):
                original_timeout = tool.timeout
                tool.timeout = self.timeout

            result = tool.execute(**kwargs)

            if result.success:
                self.logger.info(f"Tool '{tool_name}' executed successfully")
            else:
                self.logger.warning(f"Tool '{tool_name}' execution failed: {result.error}")

            return result

        except Exception as e:
            error_msg = f"Unexpected error executing tool '{tool_name}': {str(e)}"
            self.logger.error(error_msg)
            return ToolResult(False, error=error_msg)

        finally:
            if hasattr(tool, 'timeout') and 'original_timeout' in locals():
                tool.timeout = original_timeout