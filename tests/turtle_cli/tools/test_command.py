import pytest
import subprocess
import os
import tempfile
from unittest.mock import patch, MagicMock
from turtle_cli.tools.command import CommandExecutor, CommandResult, execute_command


class TestCommandResult:
    
    def test_command_result_creation(self):
        result = CommandResult(
            stdout="output",
            stderr="error",
            exit_code=0,
            timed_out=False
        )
        assert result.stdout == "output"
        assert result.stderr == "error"
        assert result.exit_code == 0
        assert result.timed_out is False


class TestCommandExecutor:
    
    def test_init_with_defaults(self):
        executor = CommandExecutor()
        assert executor.working_dir is None
        assert executor.timeout == 30
    
    def test_init_with_custom_values(self):
        executor = CommandExecutor(working_dir="/tmp", timeout=60)
        assert executor.working_dir == "/tmp"
        assert executor.timeout == 60
    
    def test_execute_successful_command_with_shell(self):
        executor = CommandExecutor()
        result = executor.execute("echo 'hello world'", shell=True)
        
        assert "hello world" in result.stdout
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.timed_out is False
    
    def test_execute_successful_command_without_shell(self):
        executor = CommandExecutor()
        result = executor.execute("echo hello", shell=False)
        
        assert "hello" in result.stdout
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.timed_out is False
    
    def test_execute_command_with_stderr(self):
        executor = CommandExecutor()
        result = executor.execute("python -c \"import sys; sys.stderr.write('error message')\"", shell=True)
        
        assert "error message" in result.stderr
        assert result.exit_code == 0
        assert result.timed_out is False
    
    def test_execute_command_with_nonzero_exit_code(self):
        executor = CommandExecutor()
        result = executor.execute("python -c \"import sys; sys.exit(1)\"", shell=True)
        
        assert result.exit_code == 1
        assert result.timed_out is False
    
    def test_execute_with_working_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = CommandExecutor(working_dir=tmpdir)
            result = executor.execute("python -c \"import os; print(os.getcwd())\"", shell=True)
            
            assert os.path.normpath(tmpdir) in os.path.normpath(result.stdout)
            assert result.exit_code == 0
    
    def test_execute_with_custom_env(self):
        executor = CommandExecutor()
        custom_env = os.environ.copy()
        custom_env['TEST_VAR'] = 'test_value'
        
        result = executor.execute("python -c \"import os; print(os.environ.get('TEST_VAR', ''))\"", env=custom_env, shell=True)
        
        assert "test_value" in result.stdout
        assert result.exit_code == 0
    
    def test_execute_with_timeout(self):
        executor = CommandExecutor(timeout=1)
        result = executor.execute("sleep 5", shell=True)
        
        assert result.stdout == ""
        assert "timed out after 1 seconds" in result.stderr
        assert result.exit_code == -1
        assert result.timed_out is True
    
    @patch('subprocess.run')
    def test_execute_with_timeout_exception(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=30)
        
        executor = CommandExecutor(timeout=30)
        result = executor.execute("test command", shell=True)
        
        assert result.stdout == ""
        assert "Command timed out after 30 seconds" in result.stderr
        assert result.exit_code == -1
        assert result.timed_out is True
    
    @patch('subprocess.run')
    def test_execute_with_generic_exception(self, mock_run):
        mock_run.side_effect = Exception("Something went wrong")
        
        executor = CommandExecutor()
        result = executor.execute("test command", shell=True)
        
        assert result.stdout == ""
        assert "Execution error: Something went wrong" in result.stderr
        assert result.exit_code == -1
        assert result.timed_out is False
    
    @patch('subprocess.run')
    def test_execute_with_permission_error(self, mock_run):
        mock_run.side_effect = PermissionError("Permission denied")
        
        executor = CommandExecutor()
        result = executor.execute("test command", shell=True)
        
        assert result.stdout == ""
        assert "Execution error: Permission denied" in result.stderr
        assert result.exit_code == -1
        assert result.timed_out is False
    
    @patch('subprocess.run')
    def test_execute_with_file_not_found_error(self, mock_run):
        mock_run.side_effect = FileNotFoundError("Command not found")
        
        executor = CommandExecutor()
        result = executor.execute("nonexistent_command", shell=False)
        
        assert result.stdout == ""
        assert "Execution error: Command not found" in result.stderr
        assert result.exit_code == -1
        assert result.timed_out is False


class TestExecuteCommandFunction:
    
    def test_execute_command_with_defaults(self):
        result = execute_command("echo 'test'")
        
        assert "test" in result.stdout
        assert result.exit_code == 0
        assert result.timed_out is False
    
    def test_execute_command_with_working_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = execute_command("python -c \"import os; print(os.getcwd())\"", working_dir=tmpdir)
            
            assert os.path.normpath(tmpdir) in os.path.normpath(result.stdout)
            assert result.exit_code == 0
    
    def test_execute_command_with_timeout(self):
        result = execute_command("sleep 5", timeout=1)
        
        assert result.timed_out is True
        assert result.exit_code == -1
    
    def test_execute_command_with_env(self):
        custom_env = os.environ.copy()
        custom_env['CUSTOM_VAR'] = 'custom_value'
        
        result = execute_command("python -c \"import os; print(os.environ.get('CUSTOM_VAR', ''))\"", env=custom_env)
        
        assert "custom_value" in result.stdout
        assert result.exit_code == 0
    
    def test_execute_command_all_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_env = os.environ.copy()
            custom_env['TEST_VAR'] = 'value'
            
            result = execute_command(
                "python -c \"import os; print(os.environ.get('TEST_VAR', ''))\"",
                working_dir=tmpdir,
                timeout=10,
                env=custom_env
            )
            
            assert "value" in result.stdout
            assert result.exit_code == 0
            assert result.timed_out is False


class TestEdgeCases:
    
    def test_empty_command_output(self):
        executor = CommandExecutor()
        result = executor.execute("python -c \"pass\"", shell=True)
        
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == 0
    
    def test_command_with_special_characters(self):
        executor = CommandExecutor()
        result = executor.execute("echo 'special !@#$%^&*()'", shell=True)
        
        assert result.exit_code == 0
        assert result.timed_out is False
    
    def test_multiline_output(self):
        executor = CommandExecutor()
        result = executor.execute("python -c \"print('line1\\nline2\\nline3')\"", shell=True)
        
        assert "line1" in result.stdout
        assert "line2" in result.stdout
        assert "line3" in result.stdout
        assert result.exit_code == 0
    
    def test_command_with_pipes(self):
        executor = CommandExecutor()
        result = executor.execute("python -c \"print('hello world')\" | python -c \"import sys; print([line for line in sys.stdin if 'hello' in line][0].strip())\"", shell=True)
        
        assert "hello" in result.stdout.lower()
        assert result.exit_code == 0
    
    def test_very_short_timeout(self):
        executor = CommandExecutor(timeout=0.001)
        result = executor.execute("sleep 1", shell=True)
        
        assert result.timed_out is True
        assert result.exit_code == -1
