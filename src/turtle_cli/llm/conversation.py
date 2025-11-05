import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
import tiktoken

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM client interface"""
    def chat(self, messages: List[Dict[str, str]]) -> str:
        ...


class ConversationManager:
    """
    Manages multi-turn conversations with context window management,
    persistence, and system prompt handling.
    """

    def __init__(
        self,
        system_prompt: Optional[str],
        max_context_tokens: int,
        model_name: str,
    ):
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.model_name = model_name
        self.messages: List[Dict[str, str]] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "turn_count": 0,
        }

        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"Model {model_name} not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")

        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

        logger.info(
            f"ConversationManager initialized with max_tokens={max_context_tokens}"
        )

    def add_message(self, role: str, content: str) -> None:
        if role not in ["system", "user", "assistant", "tool"]:
            raise ValueError(f"Invalid role: {role}")

        if not content:
            raise ValueError("Message content cannot be empty")

        self.messages.append({"role": role, "content": content})
        self.metadata["updated_at"] = datetime.now().isoformat()

        if role == "user":
            self.metadata["turn_count"] += 1

        logger.debug(f"Added {role} message ({len(content)} chars)")

    def get_messages(self, include_system: bool) -> List[Dict[str, str]]:
        if include_system:
            return self.messages.copy()
        return [msg for msg in self.messages if msg["role"] != "system"]

    def count_tokens(self, messages: Optional[List[Dict[str, str]]]) -> int:
        if messages is None:
            messages = self.messages

        total_tokens = 0
        for message in messages:
            total_tokens += len(self.encoding.encode(message["content"]))
            total_tokens += 4

        total_tokens += 2
        return total_tokens

    def truncate_context(self, target_tokens: Optional[int], llm_client: LLMClient) -> int:
        if target_tokens is None:
            target_tokens = self.max_context_tokens
        
        current_tokens = self.count_tokens(None)
        
        if current_tokens <= target_tokens:
            logger.debug(f"Context within limits: {current_tokens}/{target_tokens} tokens")
            return 0
        
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        conversation_messages = [msg for msg in self.messages if msg["role"] != "system"]
        
        split_index = 0
        for i in range(len(conversation_messages)):
            remaining = conversation_messages[i:]
            if self.count_tokens(system_messages + remaining) <= target_tokens:
                split_index = i
                break
        
        if split_index == 0:
            raise RuntimeError(
                f"Cannot fit conversation within token limit ({target_tokens} tokens). "
                f"This indicates a configuration issue: either increase max_context_tokens, "
                f"reduce message sizes, or check for abnormally large messages."
            )
        
        messages_to_summarize = conversation_messages[:split_index]
        remaining_messages = conversation_messages[split_index:]
        
        summary_text = self._create_ai_summary(messages_to_summarize, llm_client)
        summary_message = {"role": "user", "content": f"[Context Summary]: {summary_text}"}
        
        self.messages = system_messages + [summary_message] + remaining_messages
        
        logger.info(
            f"Summarized {len(messages_to_summarize)} messages. "
            f"New token count: {self.count_tokens(None)}/{target_tokens}"
        )
        
        return len(messages_to_summarize)

    def _create_ai_summary(self, messages: List[Dict[str, str]], llm_client: LLMClient) -> str:
        conversation_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in messages
        ])
        
        summary_prompt = [
            {"role": "system", "content": "Summarize the following conversation concisely in one paragraph."},
            {"role": "user", "content": conversation_text}
        ]
        
        summary = llm_client.chat(summary_prompt)
        return summary

    def prepare_messages_for_api(self, reserve_tokens: int, llm_client: LLMClient) -> List[Dict[str, str]]:
        target_tokens = self.max_context_tokens - reserve_tokens
        self.truncate_context(target_tokens, llm_client)
        return self.get_messages(True)

    def set_system_prompt(self, prompt: str, replace: bool) -> None:
        if not prompt:
            raise ValueError("System prompt cannot be empty")

        if replace:
            self.messages = [msg for msg in self.messages if msg["role"] != "system"]

        self.messages.insert(0, {"role": "system", "content": prompt})
        self.system_prompt = prompt
        logger.info("System prompt updated")

    def get_system_prompt(self) -> Optional[str]:
        for msg in self.messages:
            if msg["role"] == "system":
                return msg["content"]
        return None

    def reset(self, keep_system_prompt: bool) -> None:
        if keep_system_prompt and self.system_prompt:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        else:
            self.messages = []
            self.system_prompt = None

        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "turn_count": 0,
        }

        logger.info("Conversation reset")

    def save(self, filepath: Path | str) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        conversation_data = {
            "system_prompt": self.system_prompt,
            "max_context_tokens": self.max_context_tokens,
            "model_name": self.model_name,
            "messages": self.messages,
            "metadata": self.metadata,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Conversation saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path | str) -> "ConversationManager":
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Conversation file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        manager = cls(
            system_prompt=data.get("system_prompt"),
            max_context_tokens=data.get("max_context_tokens"),
            model_name=data.get("model_name"),
        )

        manager.messages = data.get("messages", [])
        manager.metadata = data.get("metadata", manager.metadata)

        logger.info(f"Conversation loaded from {filepath}")
        return manager

    def get_conversation_summary(self) -> Dict[str, Any]:
        return {
            "turn_count": self.metadata["turn_count"],
            "message_count": len(self.messages),
            "total_tokens": self.count_tokens(None),
            "max_tokens": self.max_context_tokens,
            "token_usage_percent": (self.count_tokens(None) / self.max_context_tokens) * 100,
            "created_at": self.metadata["created_at"],
            "updated_at": self.metadata["updated_at"],
            "has_system_prompt": self.system_prompt is not None,
        }

    def __repr__(self) -> str:
        return (
            f"ConversationManager(messages={len(self.messages)}, "
            f"tokens={self.count_tokens(None)}/{self.max_context_tokens}, "
            f"turns={self.metadata['turn_count']})"
        )


__all__ = ["ConversationManager"]