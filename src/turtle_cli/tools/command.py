import subprocess
import shlex
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


class CommandExecutor:
    """Executes shell commands with timeout handling."""
    
    def __init__(self, working_dir: Optional[str] = None, timeout: int = 30):

        self.working_dir = working_dir
        self.timeout = timeout
    
    def execute(
        self,
        command: str,
        env: Optional[Dict[str, str]] = None,
        shell: bool = True
    ) -> CommandResult:

        try:
            process = subprocess.run(
                command if shell else shlex.split(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.working_dir,
                env=env,
                shell=shell,
                timeout=self.timeout,
                text=True
            )
            
            return CommandResult(
                stdout=process.stdout,
                stderr=process.stderr,
                exit_code=process.returncode,
                timed_out=False
            )
            
        except subprocess.TimeoutExpired:
            return CommandResult(
                stdout="",
                stderr=f"Command timed out after {self.timeout} seconds",
                exit_code=-1,
                timed_out=True
            )
        
        except Exception as e:
            return CommandResult(
                stdout="",
                stderr=f"Execution error: {str(e)}",
                exit_code=-1,
                timed_out=False
            )


def execute_command(
    command: str,
    working_dir: Optional[str] = None,
    timeout: int = 30,
    env: Optional[Dict[str, str]] = None
) -> CommandResult:

    executor = CommandExecutor(working_dir=working_dir, timeout=timeout)
    return executor.execute(command, env=env)
