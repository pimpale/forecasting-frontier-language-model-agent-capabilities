import subprocess
from dataclasses import dataclass
from typing import override

from inspect_ai.tool import Tool, bash
from inspect_ai.util import sandbox

from bash_minimal.commands.BaseCommand import (
    BaseCommand,
    CallbackType,
    CommandOutput,
)
from bash_minimal.commands.CommandBlocker import CommandBlocker


@dataclass
class BashOutput(CommandOutput):
    failed: bool


class BashCommand(BaseCommand):
    def __init__(
        self,
        timeout: int,
        max_output_length: int,
        callback: CallbackType | None = None,
    ):
        self.timeout = timeout
        self.max_output_length = max_output_length
        super().__init__(
            xml_tag="bash",
            description="""To run a shell command, wrap it in <bash></bash> XML tags. Examples:
<bash>ls</bash>
<bash>python3 script.py</bash>
<bash>cat file.txt</bash>
<bash>python3 -c "
import numpy as np
print(np.sum([1, 2, 3]))
"</bash>""",
            callback=callback,
        )


    @override
    def native_tool(self) -> Tool:
        return bash(timeout=self.timeout)


    async def _run(self, content: str) -> CommandOutput:
        if CommandBlocker.should_block(content):
            return BashOutput(
                content="BASH ERROR:\nInteractive command not allowed.", failed=True
            )

        output = []

        try:
            result = await sandbox().exec(
                cmd=["bash", "--login", "-c", content], timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            return BashOutput(
                content=f"BASH ERROR:\nCommand timed out after {self.timeout} seconds.",
                failed=True,
            )
        except Exception as e:
            return BashOutput(
                content=f"BASH ERROR:\nAn unexpected error occurred: {str(e)}",
                failed=True,
            )

        if result.stdout:
            if len(result.stdout) > self.max_output_length:
                # TODO: This should be done with sandbox().write_file()
                await sandbox().write_file("command.stdout", result.stdout)
                result.stdout = (
                    result.stdout[: self.max_output_length // 2]
                    + "... STDOUT WAS TRUNCATED DUE TO LENGTH. The full output has been saved to `./command.stdout`. Use `head` and `tail` to view portions of the output ..."
                    + result.stdout[-self.max_output_length // 2 :]
                )

            output.append(f"BASH STDOUT:\n{result.stdout}")
        if result.stderr:
            if len(result.stderr) > self.max_output_length:
                await sandbox().write_file("command.stderr", result.stderr)

                result.stderr = (
                    result.stderr[: self.max_output_length // 2]
                    + "... STDERR WAS TRUNCATED DUE TO LENGTH. The full output has been saved to `./command.stderr`. Use `head` and `tail` to view portions of the output ..."
                    + result.stderr[-self.max_output_length // 2 :]
                )

            output.append(f"BASH STDERR:\n{result.stderr}")

        if not output:
            return BashOutput(
                content="BASH OUTPUT:\nCommand ran successfully with no output.",
                failed=result.returncode != 0,
            )

        return BashOutput(content="\n".join(output), failed=result.returncode != 0)
