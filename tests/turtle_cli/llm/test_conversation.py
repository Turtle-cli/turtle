import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime
from turtle_cli.llm.conversation import ConversationManager


class MockLLMClient:
    def __init__(self, summary_response="This is a test summary of the conversation."):
        self.summary_response = summary_response
        self.call_count = 0
        self.last_messages = None
    
    def chat(self, messages):
        self.call_count += 1
        self.last_messages = messages
        return self.summary_response


class TestConversationManagerInitialization:
    
    def test_basic_initialization(self):
        manager = ConversationManager(
            system_prompt=None,
            max_context_tokens=1000,
            model_name="gpt-3.5-turbo"
        )
        
        assert manager.max_context_tokens == 1000
        assert manager.model_name == "gpt-3.5-turbo"
        assert manager.system_prompt is None
        assert len(manager.messages) == 0
        assert manager.metadata["turn_count"] == 0
    
    def test_initialization_with_system_prompt(self):
        system_prompt = "You are a helpful assistant."
        manager = ConversationManager(
            system_prompt=system_prompt,
            max_context_tokens=1000,
            model_name="gpt-3.5-turbo"
        )
        
        assert manager.system_prompt == system_prompt
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "system"
        assert manager.messages[0]["content"] == system_prompt
    
    def test_initialization_with_unknown_model(self):
        manager = ConversationManager(
            system_prompt=None,
            max_context_tokens=1000,
            model_name="unknown-model"
        )
        
        assert manager.encoding is not None
    
    def test_metadata_creation(self):
        manager = ConversationManager(
            system_prompt=None,
            max_context_tokens=1000,
            model_name="gpt-3.5-turbo"
        )
        
        assert "created_at" in manager.metadata
        assert "updated_at" in manager.metadata
        assert "turn_count" in manager.metadata
        assert manager.metadata["turn_count"] == 0


class TestMessageManagement:
    
    def test_add_user_message(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello, how are you?")
        
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "user"
        assert manager.messages[0]["content"] == "Hello, how are you?"
        assert manager.metadata["turn_count"] == 1
    
    def test_add_assistant_message(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        manager.add_message("assistant", "I'm doing well, thank you!")
        
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "assistant"
        assert manager.metadata["turn_count"] == 0
    
    def test_add_multiple_messages(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there!")
        manager.add_message("user", "How are you?")
        manager.add_message("assistant", "I'm great!")
        
        assert len(manager.messages) == 4
        assert manager.metadata["turn_count"] == 2
    
    def test_add_message_invalid_role(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        
        with pytest.raises(ValueError, match="Invalid role"):
            manager.add_message("invalid_role", "content")
    
    def test_add_message_empty_content(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            manager.add_message("user", "")
    
    def test_add_tool_message(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        manager.add_message("tool", "Tool response data")
        
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "tool"
    
    def test_metadata_updated_on_message_add(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        initial_updated_at = manager.metadata["updated_at"]
        
        import time
        time.sleep(0.01)
        
        manager.add_message("user", "Test message")
        
        assert manager.metadata["updated_at"] != initial_updated_at


class TestMessageRetrieval:
    
    def test_get_messages_with_system(self):
        manager = ConversationManager("System prompt", 1000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        
        messages = manager.get_messages(include_system=True)
        
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
    
    def test_get_messages_without_system(self):
        manager = ConversationManager("System prompt", 1000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        
        messages = manager.get_messages(include_system=False)
        
        assert len(messages) == 2
        assert all(msg["role"] != "system" for msg in messages)
    
    def test_get_messages_returns_copy(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello")
        
        messages = manager.get_messages(include_system=True)
        messages.append({"role": "user", "content": "Modified"})
        
        assert len(manager.messages) == 1


class TestTokenCounting:
    
    def test_count_tokens_empty(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        token_count = manager.count_tokens(None)
        
        assert token_count >= 0
    
    def test_count_tokens_with_messages(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello, how are you?")
        manager.add_message("assistant", "I'm doing well, thank you!")
        
        token_count = manager.count_tokens(None)
        
        assert token_count > 0
    
    def test_count_tokens_custom_messages(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        
        custom_messages = [
            {"role": "user", "content": "Test message"}
        ]
        
        token_count = manager.count_tokens(custom_messages)
        
        assert token_count > 0
    
    def test_token_count_increases_with_content(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        
        manager.add_message("user", "Hi")
        token_count_1 = manager.count_tokens(None)
        
        manager.add_message("user", "This is a much longer message with many more words")
        token_count_2 = manager.count_tokens(None)
        
        assert token_count_2 > token_count_1


class TestContextTruncation:
    
    def test_truncate_within_limits(self):
        manager = ConversationManager(None, 10000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there!")
        
        llm_client = MockLLMClient()
        removed_count = manager.truncate_context(10000, llm_client)
        
        assert removed_count == 0
        assert len(manager.messages) == 2
        assert llm_client.call_count == 0
    
    def test_truncate_exceeds_limits(self):
        manager = ConversationManager(None, 10000, "gpt-3.5-turbo")
        
        for i in range(20):
            manager.add_message("user", f"This is user message number {i} " * 50)
            manager.add_message("assistant", f"This is assistant response number {i} " * 50)
        
        llm_client = MockLLMClient()
        removed_count = manager.truncate_context(500, llm_client)
        
        assert removed_count > 0
        assert llm_client.call_count == 1
        
        has_summary = any("[Context Summary]" in msg["content"] for msg in manager.messages)
        assert has_summary
    
    def test_truncate_with_system_prompt(self):
        manager = ConversationManager("You are helpful", 10000, "gpt-3.5-turbo")
        
        for i in range(15):
            manager.add_message("user", f"Message {i} " * 50)
            manager.add_message("assistant", f"Response {i} " * 50)
        
        llm_client = MockLLMClient()
        manager.truncate_context(500, llm_client)
        
        assert manager.messages[0]["role"] == "system"
        assert manager.messages[0]["content"] == "You are helpful"
    
    def test_truncate_calls_ai_summary(self):
        manager = ConversationManager(None, 10000, "gpt-3.5-turbo")
        
        for i in range(10):
            manager.add_message("user", f"User message {i} " * 30)
            manager.add_message("assistant", f"Assistant response {i} " * 30)
        
        llm_client = MockLLMClient(summary_response="Summary of old messages")
        manager.truncate_context(300, llm_client)
        
        assert llm_client.call_count == 1
        assert llm_client.last_messages is not None
        assert llm_client.last_messages[0]["role"] == "system"
        assert "Summarize" in llm_client.last_messages[0]["content"]
    
    def test_truncate_summary_content(self):
        manager = ConversationManager(None, 10000, "gpt-3.5-turbo")
        
        for i in range(10):
            manager.add_message("user", f"Question {i} " * 30)
            manager.add_message("assistant", f"Answer {i} " * 30)
        
        custom_summary = "This is my custom summary"
        llm_client = MockLLMClient(summary_response=custom_summary)
        manager.truncate_context(300, llm_client)
        
        summary_message = next(
            (msg for msg in manager.messages if "[Context Summary]" in msg["content"]),
            None
        )
        
        assert summary_message is not None
        assert custom_summary in summary_message["content"]
    
    def test_truncate_extreme_case_single_message_too_large(self):
        manager = ConversationManager(None, 10000, "gpt-3.5-turbo")
        
        large_message = "word " * 5000
        manager.add_message("user", large_message)
        manager.add_message("user", "Small message")
        
        llm_client = MockLLMClient()
        manager.truncate_context(100, llm_client)
        
        assert len(manager.messages) >= 1
    
    def test_truncate_target_tokens_none(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        
        for i in range(10):
            manager.add_message("user", f"Message {i} " * 50)
        
        llm_client = MockLLMClient()
        manager.truncate_context(None, llm_client)
        
        token_count = manager.count_tokens(None)
        assert token_count <= 1000
    
    def test_truncate_raises_on_impossible_fit(self):
        manager = ConversationManager(None, 10000, "gpt-3.5-turbo")
        for i in range(5):
            manager.add_message("user", "word " * 200)
    
        llm_client = MockLLMClient()
        with pytest.raises(RuntimeError, match="Cannot fit conversation within token limit"):
            manager.truncate_context(50, llm_client)

class TestPrepareMessagesForAPI:
    
    def test_prepare_messages_basic(self):
        manager = ConversationManager("System", 1000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        
        llm_client = MockLLMClient()
        messages = manager.prepare_messages_for_api(100, llm_client)
        
        assert len(messages) >= 3
        assert messages[0]["role"] == "system"
    
    def test_prepare_messages_reserves_tokens(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        
        for i in range(15):
            manager.add_message("user", f"Message {i} " * 40)
        
        llm_client = MockLLMClient()
        messages = manager.prepare_messages_for_api(200, llm_client)
        
        token_count = manager.count_tokens(messages)
        assert token_count <= 800
    
    def test_prepare_messages_triggers_truncation(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        
        for i in range(20):
            manager.add_message("user", f"Message {i} " * 30)
        
        llm_client = MockLLMClient()
        messages = manager.prepare_messages_for_api(500, llm_client)
        
        assert llm_client.call_count >= 1


class TestSystemPromptManagement:
    
    def test_set_system_prompt_new(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello")
        
        manager.set_system_prompt("You are helpful", replace=False)
        
        assert manager.system_prompt == "You are helpful"
        assert manager.messages[0]["role"] == "system"
        assert len(manager.messages) == 2
    
    def test_set_system_prompt_replace(self):
        manager = ConversationManager("Old prompt", 1000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello")
        
        manager.set_system_prompt("New prompt", replace=True)
        
        assert manager.system_prompt == "New prompt"
        system_prompts = [msg for msg in manager.messages if msg["role"] == "system"]
        assert len(system_prompts) == 1
        assert system_prompts[0]["content"] == "New prompt"
    
    def test_set_system_prompt_empty(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        
        with pytest.raises(ValueError, match="System prompt cannot be empty"):
            manager.set_system_prompt("", replace=False)
    
    def test_get_system_prompt_exists(self):
        manager = ConversationManager("Test prompt", 1000, "gpt-3.5-turbo")
        
        prompt = manager.get_system_prompt()
        
        assert prompt == "Test prompt"
    
    def test_get_system_prompt_none(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        
        prompt = manager.get_system_prompt()
        
        assert prompt is None


class TestConversationReset:
    
    def test_reset_keep_system_prompt(self):
        manager = ConversationManager("System", 1000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        
        manager.reset(keep_system_prompt=True)
        
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "system"
        assert manager.system_prompt == "System"
        assert manager.metadata["turn_count"] == 0
    
    def test_reset_remove_system_prompt(self):
        manager = ConversationManager("System", 1000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello")
        
        manager.reset(keep_system_prompt=False)
        
        assert len(manager.messages) == 0
        assert manager.system_prompt is None
        assert manager.metadata["turn_count"] == 0
    
    def test_reset_clears_metadata(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello")
        manager.metadata["turn_count"] = 10
        
        old_created_at = manager.metadata["created_at"]
        
        import time
        time.sleep(0.01)
        
        manager.reset(keep_system_prompt=False)
        
        assert manager.metadata["turn_count"] == 0
        assert manager.metadata["created_at"] != old_created_at


class TestPersistence:
    
    def test_save_conversation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "conversation.json"
            
            manager = ConversationManager("System", 1000, "gpt-3.5-turbo")
            manager.add_message("user", "Hello")
            manager.add_message("assistant", "Hi there!")
            
            manager.save(filepath)
            
            assert filepath.exists()
            
            with open(filepath, "r") as f:
                data = json.load(f)
            
            assert data["system_prompt"] == "System"
            assert data["max_context_tokens"] == 1000
            assert data["model_name"] == "gpt-3.5-turbo"
            assert len(data["messages"]) == 3
    
    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "conversation.json"
            
            manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
            manager.save(filepath)
            
            assert filepath.exists()
            assert filepath.parent.exists()
    
    def test_load_conversation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "conversation.json"
            
            manager1 = ConversationManager("System", 1000, "gpt-3.5-turbo")
            manager1.add_message("user", "Hello")
            manager1.add_message("assistant", "Hi there!")
            manager1.save(filepath)
            
            manager2 = ConversationManager.load(filepath)
            
            assert manager2.system_prompt == "System"
            assert manager2.max_context_tokens == 1000
            assert manager2.model_name == "gpt-3.5-turbo"
            assert len(manager2.messages) == 3
            assert manager2.messages[1]["content"] == "Hello"
    
    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            ConversationManager.load("/nonexistent/path/conversation.json")
    
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "conversation.json"
            
            manager1 = ConversationManager("Test System", 2000, "gpt-4")
            manager1.add_message("user", "Question 1")
            manager1.add_message("assistant", "Answer 1")
            manager1.add_message("user", "Question 2")
            manager1.metadata["custom_field"] = "custom_value"
            
            manager1.save(filepath)
            manager2 = ConversationManager.load(filepath)
            
            assert manager1.system_prompt == manager2.system_prompt
            assert manager1.max_context_tokens == manager2.max_context_tokens
            assert manager1.model_name == manager2.model_name
            assert len(manager1.messages) == len(manager2.messages)
            assert manager1.metadata["turn_count"] == manager2.metadata["turn_count"]
    
    def test_save_with_unicode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "conversation.json"
            
            manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
            manager.add_message("user", "Hello 世界 🌍 café")
            
            manager.save(filepath)
            manager2 = ConversationManager.load(filepath)
            
            assert manager2.messages[0]["content"] == "Hello 世界 🌍 café"


class TestConversationSummary:
    
    def test_get_conversation_summary_empty(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        
        summary = manager.get_conversation_summary()
        
        assert summary["turn_count"] == 0
        assert summary["message_count"] == 0
        assert summary["total_tokens"] >= 0
        assert summary["max_tokens"] == 1000
        assert summary["has_system_prompt"] is False
    
    def test_get_conversation_summary_with_messages(self):
        manager = ConversationManager("System", 1000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        manager.add_message("user", "How are you?")
        
        summary = manager.get_conversation_summary()
        
        assert summary["turn_count"] == 2
        assert summary["message_count"] == 4
        assert summary["total_tokens"] > 0
        assert summary["has_system_prompt"] is True
        assert "token_usage_percent" in summary
        assert 0 <= summary["token_usage_percent"] <= 100
    
    def test_get_conversation_summary_fields(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        
        summary = manager.get_conversation_summary()
        
        required_fields = [
            "turn_count", "message_count", "total_tokens", "max_tokens",
            "token_usage_percent", "created_at", "updated_at", "has_system_prompt"
        ]
        
        for field in required_fields:
            assert field in summary


class TestRepr:
    
    def test_repr_format(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        manager.add_message("user", "Hello")
        
        repr_str = repr(manager)
        
        assert "ConversationManager" in repr_str
        assert "messages=" in repr_str
        assert "tokens=" in repr_str
        assert "turns=" in repr_str
    
    def test_repr_values(self):
        manager = ConversationManager("System", 2000, "gpt-3.5-turbo")
        manager.add_message("user", "Test")
        
        repr_str = repr(manager)
        
        assert "messages=2" in repr_str
        assert "/2000" in repr_str
        assert "turns=1" in repr_str


class TestEdgeCases:
    
    def test_very_long_message(self):
        manager = ConversationManager(None, 10000, "gpt-3.5-turbo")
        
        long_message = "word " * 10000
        manager.add_message("user", long_message)
        
        assert len(manager.messages) == 1
        token_count = manager.count_tokens(None)
        assert token_count > 0
    
    def test_special_characters_in_messages(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        
        special_message = "Test\n\t\r\x00Special: <>\"'&"
        manager.add_message("user", special_message)
        
        assert manager.messages[0]["content"] == special_message
    
    def test_empty_conversation_truncation(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        
        llm_client = MockLLMClient()
        removed = manager.truncate_context(100, llm_client)
        
        assert removed == 0
        assert llm_client.call_count == 0
    
    def test_max_context_tokens_zero(self):
        manager = ConversationManager(None, 0, "gpt-3.5-turbo")
        manager.add_message("user", "Test")
        
        token_count = manager.count_tokens(None)
        assert token_count >= 0
    
    def test_concurrent_message_additions(self):
        manager = ConversationManager(None, 1000, "gpt-3.5-turbo")
        
        for i in range(100):
            manager.add_message("user", f"Message {i}")
        
        for i in range(100):
            assert f"Message {i}" in manager.messages[i]["content"]


class TestIntegration:
    
    def test_full_conversation_flow(self):
        manager = ConversationManager(
            "You are a helpful assistant",
            2000,
            "gpt-3.5-turbo"
        )
        
        manager.add_message("user", "What is Python?")
        manager.add_message("assistant", "Python is a programming language.")
        manager.add_message("user", "Tell me more.")
        manager.add_message("assistant", "It's known for simplicity.")
        
        assert manager.metadata["turn_count"] == 2
        assert len(manager.messages) == 5
        
        llm_client = MockLLMClient()
        messages = manager.prepare_messages_for_api(500, llm_client)
        
        assert len(messages) > 0
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "conv.json"
            manager.save(filepath)
            
            manager2 = ConversationManager.load(filepath)
            assert len(manager2.messages) == len(manager.messages)
    
    def test_conversation_with_truncation_and_persistence(self):
        manager = ConversationManager("System", 1000, "gpt-3.5-turbo")
        
        for i in range(20):
            manager.add_message("user", f"Question {i} " * 20)
            manager.add_message("assistant", f"Answer {i} " * 20)
        
        llm_client = MockLLMClient()
        manager.truncate_context(500, llm_client)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "truncated.json"
            manager.save(filepath)
            
            manager2 = ConversationManager.load(filepath)
            
            has_summary = any(
                "[Context Summary]" in msg["content"] 
                for msg in manager2.messages
            )
            assert has_summary
    
    def test_system_prompt_update_mid_conversation(self):
        manager = ConversationManager("Original", 1000, "gpt-3.5-turbo")
        
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        
        manager.set_system_prompt("Updated prompt", replace=True)
        
        manager.add_message("user", "Continue")
        
        assert manager.get_system_prompt() == "Updated prompt"
        assert len([msg for msg in manager.messages if msg["role"] == "system"]) == 1
        