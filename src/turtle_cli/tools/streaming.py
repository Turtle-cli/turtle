import json
import logging
from typing import Any, Dict, Generator, List, Optional, Union
from dataclasses import dataclass
from ..llm.client import LLMClient
from ..llm.conversation import ConversationManager
from .protocol import ToolRegistry
from .parser import ToolCallParser, ParsedToolCall
from .executor import ToolExecutor
from .formatter import LiteLLMFormatter

logger = logging.getLogger(__name__)


@dataclass
class StreamBuffer:
    content: str = ""
    tool_calls: List[ParsedToolCall] = None
    is_complete: bool = False


class StreamingToolOrchestrator:
    """
    Handles streaming LLM responses with real-time tool call detection and execution.
    Interrupts streaming when tool calls are detected, executes tools, and resumes streaming.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        conversation_manager: ConversationManager,
        tool_registry: ToolRegistry,
        max_iterations: int = 10
    ):
        self.llm_client = llm_client
        self.conversation_manager = conversation_manager
        self.tool_executor = ToolExecutor(tool_registry)
        self.max_iterations = max_iterations
        self.iteration_count = 0

        logger.info(f"StreamingToolOrchestrator initialized with max_iterations={max_iterations}")

    def execute_streaming_loop(self, user_input: str) -> Generator[str, None, None]:
        self.iteration_count = 0
        self.conversation_manager.add_message("user", user_input)

        logger.info("Starting streaming tool orchestration loop")

        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            logger.debug(f"Streaming loop iteration {self.iteration_count}/{self.max_iterations}")

            messages = self.conversation_manager.prepare_messages_for_api(
                reserve_tokens=1000,
                llm_client=self.llm_client
            )

            yielded_content = ""
            stream_buffer = StreamBuffer()

            try:
                stream_gen = self.llm_client.stream(
                    messages=messages,
                    tools=self.tool_executor.registry.export_openai_format()
                )

                for chunk in self._process_stream_with_tool_detection(stream_gen, stream_buffer):
                    if chunk:
                        yielded_content += chunk
                        yield chunk

                if stream_buffer.tool_calls:
                    logger.info(f"Tool calls detected, executing {len(stream_buffer.tool_calls)} tools")

                    if yielded_content.strip():
                        self.conversation_manager.add_message("assistant", yielded_content)

                    self._execute_tool_calls(stream_buffer.tool_calls)
                    continue
                else:
                    if yielded_content.strip():
                        self.conversation_manager.add_message("assistant", yielded_content)
                    logger.info("No tool calls found, ending streaming loop")
                    return

            except Exception as e:
                logger.error(f"Error in streaming loop iteration {self.iteration_count}: {e}")
                yield f"Error: {str(e)}"
                return

        logger.warning(f"Maximum streaming iterations ({self.max_iterations}) reached")

    def _process_stream_with_tool_detection(
        self,
        stream_gen: Generator[str, None, None],
        buffer: StreamBuffer
    ) -> Generator[str, None, None]:
        accumulated_content = ""

        for chunk in stream_gen:
            accumulated_content += chunk
            buffer.content = accumulated_content

            tool_calls = self._detect_partial_tool_calls(accumulated_content)

            if tool_calls:
                logger.debug("Tool calls detected in stream, interrupting")
                buffer.tool_calls = tool_calls
                buffer.is_complete = True

                content_before_tools = self._extract_content_before_tools(accumulated_content)
                if content_before_tools:
                    yield content_before_tools
                return

            yield chunk

    def _detect_partial_tool_calls(self, content: str) -> Optional[List[ParsedToolCall]]:
        try:
            if "<|tool_call|>" in content or '"tool_calls"' in content:
                mock_response = {
                    "choices": [{
                        "message": {
                            "content": content,
                            "tool_calls": self._extract_tool_calls_from_content(content)
                        }
                    }]
                }

                return ToolCallParser.parse_tool_calls(mock_response)
        except Exception as e:
            logger.debug(f"Error detecting tool calls in partial content: {e}")

        return None

    def _extract_tool_calls_from_content(self, content: str) -> Optional[List[Dict[str, Any]]]:
        try:
            if '"tool_calls":' in content:
                start = content.find('"tool_calls":')
                if start != -1:
                    bracket_count = 0
                    in_array = False
                    tool_calls_start = None

                    for i, char in enumerate(content[start:], start):
                        if char == '[' and not in_array:
                            tool_calls_start = i
                            bracket_count = 1
                            in_array = True
                        elif char == '[' and in_array:
                            bracket_count += 1
                        elif char == ']' and in_array:
                            bracket_count -= 1
                            if bracket_count == 0:
                                tool_calls_json = content[tool_calls_start:i+1]
                                return json.loads(tool_calls_json)

            if content.strip().startswith('[{') and '"function"' in content:
                return json.loads(content.strip())

        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def _extract_content_before_tools(self, content: str) -> str:
        tool_markers = ["<|tool_call|>", '"tool_calls":', '[{"id":']

        for marker in tool_markers:
            if marker in content:
                return content[:content.find(marker)].strip()

        return content.strip()

    def _execute_tool_calls(self, tool_calls: List[ParsedToolCall]) -> None:
        logger.info(f"Executing {len(tool_calls)} tool calls in streaming context")

        for tool_call in tool_calls:
            result = self.tool_executor.execute(
                tool_call.function_name,
                **tool_call.arguments
            )

            formatted_response = LiteLLMFormatter.format_tool_response(
                tool_call.id,
                result,
                tool_call.function_name
            )

            self.conversation_manager.add_message(
                "tool",
                formatted_response["content"]
            )

    def reset_iteration_count(self) -> None:
        self.iteration_count = 0
        logger.debug("Streaming iteration count reset")

    def get_conversation_state(self) -> Dict[str, Any]:
        return {
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "conversation_summary": self.conversation_manager.get_conversation_summary(),
            "mode": "streaming"
        }
