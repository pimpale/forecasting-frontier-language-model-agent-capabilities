from typing import Optional, override

from inspect_ai.tool import Tool, ToolDef

from bash_minimal.commands.BaseCommand import (
    BaseCommand,
    CallbackType,
    CommandOutput,
)


class SubmitCommand(BaseCommand):
    def __init__(self, callback: Optional[CallbackType] = None):
        super().__init__(
            xml_tag="submit",
            description="To submit an answer, wrap it in <submit></submit> XML tags. Example: <submit>YOUR ANSWER HERE</submit>",
            callback=callback,
        )

    @override
    def native_tool(self) -> ToolDef:
        async def submit(answer: str):
            assert self.callback is not None
            self.callback(answer, "Successfully submitted answer.")
            return "Successfully submitted answer."
        
        return ToolDef(
            tool=submit,
            name="submit",
            description="A function to submit answers.", 
            parameters={
                "answer": "the answer",
            }
        )
        

    async def _run(self, content: str) -> CommandOutput:
        content = content.strip()
        return CommandOutput("Successfully submitted answer.")
