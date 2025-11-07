import logging
from typing import Any, Dict, List, Optional
from ..llm.client import LLMClient
from ..llm.conversation import ConversationManager
from .protocol import ToolRegistry
from .parser import ToolCallParser, ParsedToolCall
from .executor import ToolExecutor
from .formatter import LiteLLMFormatter

logger = logging.getLogger(__name__)


class ToolOrchestrator:
    """
    Orchestrates the tool execution loop: LLM -> parse -> execute -> LLM
    Manages conversation state and integrates with ConversationManager
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

        logger.info(f"ToolOrchestrator initialized with max_iterations={max_iterations}")

    def execute_loop(self, user_input: str) -> str:
        self.iteration_count = 0
        self.conversation_manager.add_message("user", user_input)

        logger.info("Starting tool orchestration loop")

        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            logger.debug(f"Loop iteration {self.iteration_count}/{self.max_iterations}")

            messages = self.conversation_manager.prepare_messages_for_api(
                reserve_tokens=1000,
                llm_client=self.llm_client
            )

            response = self.llm_client.chat(
                messages=messages,
                tools=self.tool_executor.registry.export_openai_format()
            )

            tool_calls = ToolCallParser.parse_tool_calls(response)

            if not tool_calls:
                logger.info("No tool calls found, ending loop")
                assistant_content = self._extract_assistant_content(response)
                self.conversation_manager.add_message("assistant", assistant_content)
                return assistant_content

            self._execute_tool_calls(tool_calls, response)

        logger.warning(f"Maximum iterations ({self.max_iterations}) reached")
        return "Maximum iteration limit reached"

    def _execute_tool_calls(self, tool_calls: List[ParsedToolCall], llm_response: Any) -> None:
        logger.info(f"Executing {len(tool_calls)} tool calls")

        assistant_content = self._extract_assistant_content(llm_response)
        if assistant_content:
            self.conversation_manager.add_message("assistant", assistant_content)

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

    def _extract_assistant_content(self, response: Any) -> str:
        if isinstance(response, dict):
            if "choices" in response and response["choices"]:
                message = response["choices"][0].get("message", {})
                return message.get("content", "")
        elif hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content or ""

        return ""

    def reset_iteration_count(self) -> None:
        self.iteration_count = 0
        logger.debug("Iteration count reset")

    def get_conversation_state(self) -> Dict[str, Any]:
        return {
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "conversation_summary": self.conversation_manager.get_conversation_summary()
        }