import json
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from turtle_cli.tools.streaming import StreamingToolOrchestrator, StreamBuffer
from turtle_cli.tools.parser import ParsedToolCall


class TestStreamBuffer:
    def test_stream_buffer_default_init(self):
        buffer = StreamBuffer()
        assert buffer.content == ""
        assert buffer.tool_calls is None
        assert buffer.is_complete is False

    def test_stream_buffer_custom_init(self):
        tool_calls = [Mock()]
        buffer = StreamBuffer(content="test", tool_calls=tool_calls, is_complete=True)
        assert buffer.content == "test"
        assert buffer.tool_calls == tool_calls
        assert buffer.is_complete is True


class TestStreamingToolOrchestrator:
    @pytest.fixture
    def mock_llm_client(self):
        return Mock()

    @pytest.fixture
    def mock_conversation_manager(self):
        manager = Mock()
        manager.prepare_messages_for_api.return_value = []
        return manager

    @pytest.fixture
    def mock_tool_registry(self):
        registry = Mock()
        registry.export_openai_format.return_value = []
        return registry

    @pytest.fixture
    def orchestrator(self, mock_llm_client, mock_conversation_manager, mock_tool_registry):
        return StreamingToolOrchestrator(
            mock_llm_client,
            mock_conversation_manager,
            mock_tool_registry,
            max_iterations=5
        )

    @patch('turtle_cli.tools.streaming.logger')
    def test_init_logs_max_iterations(self, mock_logger, mock_llm_client, mock_conversation_manager, mock_tool_registry):
        orchestrator = StreamingToolOrchestrator(
            mock_llm_client,
            mock_conversation_manager,
            mock_tool_registry,
            max_iterations=10
        )
        mock_logger.info.assert_called_with("StreamingToolOrchestrator initialized with max_iterations=10")
        assert orchestrator.max_iterations == 10
        assert orchestrator.iteration_count == 0

    @patch('turtle_cli.tools.streaming.logger')
    def test_execute_streaming_loop_no_tool_calls(self, mock_logger, orchestrator):
        orchestrator.llm_client.stream.return_value = ["chunk1", "chunk2"]

        with patch.object(orchestrator, '_process_stream_with_tool_detection') as mock_process:
            mock_process.return_value = ["chunk1", "chunk2"]

            result = list(orchestrator.execute_streaming_loop("test input"))

            assert result == ["chunk1", "chunk2"]
            orchestrator.conversation_manager.add_message.assert_has_calls([
                call("user", "test input"),
                call("assistant", "chunk1chunk2")
            ])
            mock_logger.info.assert_any_call("Starting streaming tool orchestration loop")
            mock_logger.info.assert_any_call("No tool calls found, ending streaming loop")

    @patch('turtle_cli.tools.streaming.logger')
    def test_execute_streaming_loop_with_tool_calls(self, mock_logger, orchestrator):
        mock_tool_call = ParsedToolCall(id="call_1", function_name="test_func", arguments={})

        # First call returns stream with tool calls, second call returns empty (normal end)
        orchestrator.llm_client.stream.side_effect = [["chunk1"], []]

        with patch.object(orchestrator, '_process_stream_with_tool_detection') as mock_process, \
             patch.object(orchestrator, '_execute_tool_calls') as mock_execute:

            def mock_process_side_effect(gen, buf):
                chunks = list(gen)
                # Only set tool calls on first iteration
                if chunks and chunks[0] == "chunk1":
                    buf.tool_calls = [mock_tool_call]
                else:
                    buf.tool_calls = None
                return chunks

            mock_process.side_effect = mock_process_side_effect

            result = list(orchestrator.execute_streaming_loop("test input"))

            assert result == ["chunk1"]
            mock_execute.assert_called_once_with([mock_tool_call])
            mock_logger.info.assert_any_call("Tool calls detected, executing 1 tools")

    @patch('turtle_cli.tools.streaming.logger')
    def test_execute_streaming_loop_max_iterations_reached(self, mock_logger, orchestrator):
        orchestrator.max_iterations = 1
        orchestrator.llm_client.stream.return_value = ["chunk"]

        with patch.object(orchestrator, '_process_stream_with_tool_detection') as mock_process, \
             patch.object(orchestrator, '_execute_tool_calls'):

            mock_tool_call = ParsedToolCall(id="call_1", function_name="test_func", arguments={})

            def mock_process_side_effect(gen, buf):
                chunks = list(gen)
                buf.tool_calls = [mock_tool_call]
                return chunks

            mock_process.side_effect = mock_process_side_effect

            result = list(orchestrator.execute_streaming_loop("test input"))

            mock_logger.warning.assert_called_with("Maximum streaming iterations (1) reached")

    @patch('turtle_cli.tools.streaming.logger')
    def test_execute_streaming_loop_exception_handling(self, mock_logger, orchestrator):
        orchestrator.llm_client.stream.side_effect = Exception("Stream error")

        result = list(orchestrator.execute_streaming_loop("test input"))

        assert result == ["Error: Stream error"]
        mock_logger.error.assert_called_with("Error in streaming loop iteration 1: Stream error")

    def test_execute_streaming_loop_empty_content(self, orchestrator):
        orchestrator.llm_client.stream.return_value = ["", "  ", "\n"]

        with patch.object(orchestrator, '_process_stream_with_tool_detection') as mock_process:
            mock_process.return_value = ["", "  ", "\n"]

            result = list(orchestrator.execute_streaming_loop("test input"))

            orchestrator.conversation_manager.add_message.assert_called_once_with("user", "test input")

    def test_process_stream_with_tool_detection_no_tools(self, orchestrator):
        stream_gen = ["chunk1", "chunk2"]
        buffer = StreamBuffer()

        with patch.object(orchestrator, '_detect_partial_tool_calls', return_value=None):
            result = list(orchestrator._process_stream_with_tool_detection(stream_gen, buffer))

            assert result == ["chunk1", "chunk2"]
            assert buffer.content == "chunk1chunk2"
            assert buffer.tool_calls is None
            assert not buffer.is_complete

    @patch('turtle_cli.tools.streaming.logger')
    def test_process_stream_with_tool_detection_with_tools(self, mock_logger, orchestrator):
        mock_tool_call = ParsedToolCall(id="call_1", function_name="test_func", arguments={})
        stream_gen = ["chunk1", "chunk2"]
        buffer = StreamBuffer()

        with patch.object(orchestrator, '_detect_partial_tool_calls') as mock_detect, \
             patch.object(orchestrator, '_extract_content_before_tools', return_value="content"):

            mock_detect.side_effect = [None, [mock_tool_call]]

            result = list(orchestrator._process_stream_with_tool_detection(stream_gen, buffer))

            assert result == ["chunk1", "content"]
            assert buffer.tool_calls == [mock_tool_call]
            assert buffer.is_complete
            mock_logger.debug.assert_called_with("Tool calls detected in stream, interrupting")

    def test_process_stream_with_tool_detection_no_content_before_tools(self, orchestrator):
        mock_tool_call = ParsedToolCall(id="call_1", function_name="test_func", arguments={})
        stream_gen = ["chunk1"]
        buffer = StreamBuffer()

        with patch.object(orchestrator, '_detect_partial_tool_calls', return_value=[mock_tool_call]), \
             patch.object(orchestrator, '_extract_content_before_tools', return_value=""):

            result = list(orchestrator._process_stream_with_tool_detection(stream_gen, buffer))

            assert result == []
            assert buffer.tool_calls == [mock_tool_call]

    def test_detect_partial_tool_calls_with_tool_marker(self, orchestrator):
        content = "Some text <|tool_call|> more text"

        with patch.object(orchestrator, '_extract_tool_calls_from_content', return_value=[{"id": "test"}]), \
             patch('turtle_cli.tools.streaming.ToolCallParser.parse_tool_calls') as mock_parse:

            mock_parse.return_value = [ParsedToolCall(id="test", function_name="func", arguments={})]

            result = orchestrator._detect_partial_tool_calls(content)

            assert len(result) == 1
            mock_parse.assert_called_once()

    def test_detect_partial_tool_calls_with_json_marker(self, orchestrator):
        content = 'Some text "tool_calls" more text'

        with patch.object(orchestrator, '_extract_tool_calls_from_content', return_value=[{"id": "test"}]), \
             patch('turtle_cli.tools.streaming.ToolCallParser.parse_tool_calls') as mock_parse:

            mock_parse.return_value = [ParsedToolCall(id="test", function_name="func", arguments={})]

            result = orchestrator._detect_partial_tool_calls(content)

            assert len(result) == 1

    def test_detect_partial_tool_calls_no_markers(self, orchestrator):
        content = "Regular text without markers"

        result = orchestrator._detect_partial_tool_calls(content)

        assert result is None

    @patch('turtle_cli.tools.streaming.logger')
    def test_detect_partial_tool_calls_exception(self, mock_logger, orchestrator):
        content = "Some text <|tool_call|> more text"

        with patch.object(orchestrator, '_extract_tool_calls_from_content', side_effect=Exception("Parse error")):

            result = orchestrator._detect_partial_tool_calls(content)

            assert result is None
            mock_logger.debug.assert_called_with("Error detecting tool calls in partial content: Parse error")

    def test_extract_tool_calls_from_content_with_tool_calls_key(self, orchestrator):
        content = 'Some text "tool_calls": [{"id": "test", "function": {"name": "func"}}] more'

        result = orchestrator._extract_tool_calls_from_content(content)

        assert result == [{"id": "test", "function": {"name": "func"}}]

    def test_extract_tool_calls_from_content_nested_brackets(self, orchestrator):
        content = 'Some text "tool_calls": [{"id": "test", "data": [1, 2, 3]}] more'

        result = orchestrator._extract_tool_calls_from_content(content)

        assert result == [{"id": "test", "data": [1, 2, 3]}]

    def test_extract_tool_calls_from_content_direct_json(self, orchestrator):
        content = '[{"id": "test", "function": {"name": "func"}}]'

        result = orchestrator._extract_tool_calls_from_content(content)

        assert result == [{"id": "test", "function": {"name": "func"}}]

    def test_extract_tool_calls_from_content_invalid_json(self, orchestrator):
        content = 'Some text "tool_calls": [invalid json] more'

        result = orchestrator._extract_tool_calls_from_content(content)

        assert result is None

    def test_extract_tool_calls_from_content_no_closing_bracket(self, orchestrator):
        content = 'Some text "tool_calls": [{"id": "test"'

        result = orchestrator._extract_tool_calls_from_content(content)

        assert result is None

    def test_extract_tool_calls_from_content_no_markers(self, orchestrator):
        content = "Regular text without tool calls"

        result = orchestrator._extract_tool_calls_from_content(content)

        assert result is None

    def test_extract_content_before_tools_tool_call_marker(self, orchestrator):
        content = "Some content <|tool_call|> tool data"

        result = orchestrator._extract_content_before_tools(content)

        assert result == "Some content"

    def test_extract_content_before_tools_json_marker(self, orchestrator):
        content = 'Some content "tool_calls": [data]'

        result = orchestrator._extract_content_before_tools(content)

        assert result == "Some content"

    def test_extract_content_before_tools_id_marker(self, orchestrator):
        content = 'Some content [{"id": "test"}]'

        result = orchestrator._extract_content_before_tools(content)

        assert result == "Some content"

    def test_extract_content_before_tools_no_markers(self, orchestrator):
        content = "Regular content without markers"

        result = orchestrator._extract_content_before_tools(content)

        assert result == "Regular content without markers"

    def test_extract_content_before_tools_whitespace(self, orchestrator):
        content = "  Content with spaces  <|tool_call|> tool"

        result = orchestrator._extract_content_before_tools(content)

        assert result == "Content with spaces"

    @patch('turtle_cli.tools.streaming.logger')
    @patch('turtle_cli.tools.streaming.LiteLLMFormatter')
    def test_execute_tool_calls(self, mock_formatter, mock_logger, orchestrator):
        mock_tool_call = ParsedToolCall(id="call_1", function_name="test_func", arguments={"key": "value"})

        with patch.object(orchestrator.tool_executor, 'execute', return_value="tool_result") as mock_execute:
            mock_formatter.format_tool_response.return_value = {"content": "formatted_response"}

            orchestrator._execute_tool_calls([mock_tool_call])

            mock_execute.assert_called_once_with("test_func", key="value")
            mock_formatter.format_tool_response.assert_called_once_with("call_1", "tool_result", "test_func")
            orchestrator.conversation_manager.add_message.assert_called_once_with("tool", "formatted_response")
            mock_logger.info.assert_called_with("Executing 1 tool calls in streaming context")

    @patch('turtle_cli.tools.streaming.logger')
    def test_reset_iteration_count(self, mock_logger, orchestrator):
        orchestrator.iteration_count = 5

        orchestrator.reset_iteration_count()

        assert orchestrator.iteration_count == 0
        mock_logger.debug.assert_called_with("Streaming iteration count reset")

    def test_get_conversation_state(self, orchestrator):
        orchestrator.iteration_count = 3
        orchestrator.conversation_manager.get_conversation_summary.return_value = "summary"

        state = orchestrator.get_conversation_state()

        assert state == {
            "iteration_count": 3,
            "max_iterations": 5,
            "conversation_summary": "summary",
            "mode": "streaming"
        }

    def test_extract_tool_calls_bracket_tracking_edge_cases(self, orchestrator):
        content = 'text "tool_calls": [{"nested": {"deep": [1, 2]}, "id": "test"}] end'

        result = orchestrator._extract_tool_calls_from_content(content)

        assert result == [{"nested": {"deep": [1, 2]}, "id": "test"}]

    def test_extract_tool_calls_multiple_arrays_in_content(self, orchestrator):
        content = 'other_array: [1, 2, 3] "tool_calls": [{"id": "test"}] more_data'

        result = orchestrator._extract_tool_calls_from_content(content)

        assert result == [{"id": "test"}]

    @patch('turtle_cli.tools.streaming.ToolCallParser.parse_tool_calls')
    def test_detect_partial_tool_calls_parser_returns_none(self, mock_parse, orchestrator):
        mock_parse.return_value = None
        content = "Some text <|tool_call|> more text"

        with patch.object(orchestrator, '_extract_tool_calls_from_content', return_value=[{"id": "test"}]):
            result = orchestrator._detect_partial_tool_calls(content)

            assert result is None

    def test_multiple_tool_calls_execution(self, orchestrator):
        mock_tool_call_1 = ParsedToolCall(id="call_1", function_name="func1", arguments={"a": 1})
        mock_tool_call_2 = ParsedToolCall(id="call_2", function_name="func2", arguments={"b": 2})

        with patch.object(orchestrator.tool_executor, 'execute', side_effect=["result1", "result2"]) as mock_execute:
            with patch('turtle_cli.tools.streaming.LiteLLMFormatter') as mock_formatter:
                mock_formatter.format_tool_response.side_effect = [
                    {"content": "response1"},
                    {"content": "response2"}
                ]

                orchestrator._execute_tool_calls([mock_tool_call_1, mock_tool_call_2])

                assert mock_execute.call_count == 2
                mock_execute.assert_has_calls([
                    call("func1", a=1),
                    call("func2", b=2)
                ])

                assert orchestrator.conversation_manager.add_message.call_count == 2
                orchestrator.conversation_manager.add_message.assert_has_calls([
                    call("tool", "response1"),
                    call("tool", "response2")
                ])

    @patch('turtle_cli.tools.streaming.logger')
    def test_debug_logging_in_streaming_loop(self, mock_logger, orchestrator):
        orchestrator.llm_client.stream.return_value = ["chunk"]

        with patch.object(orchestrator, '_process_stream_with_tool_detection') as mock_process:
            mock_process.return_value = ["chunk"]
            list(orchestrator.execute_streaming_loop("test"))

            mock_logger.debug.assert_called_with("Streaming loop iteration 1/5")

    def test_streaming_loop_continues_with_tool_calls_empty_content(self, orchestrator):
        mock_tool_call = ParsedToolCall(id="call_1", function_name="test_func", arguments={})
        orchestrator.llm_client.stream.return_value = [""]

        with patch.object(orchestrator, '_process_stream_with_tool_detection') as mock_process, \
             patch.object(orchestrator, '_execute_tool_calls'):

            def mock_process_side_effect(gen, buf):
                chunks = list(gen)
                buf.tool_calls = [mock_tool_call]
                return chunks

            mock_process.side_effect = mock_process_side_effect

            list(orchestrator.execute_streaming_loop("test input"))

            calls = orchestrator.conversation_manager.add_message.call_args_list
            user_call = calls[0]
            assert user_call == call("user", "test input")
            assert len(calls) == 1
