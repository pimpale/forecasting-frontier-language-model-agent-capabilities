import os
import uuid

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
    call_tools,
    get_model,
)
from inspect_ai.solver import TaskState
from inspect_ai.solver._util import append_system_message
from openai import BadRequestError, UnprocessableEntityError

from bash_minimal.commands.BaseCommand import BaseCommand
from bash_minimal.commands.CommandHandler import CommandHandler
from bash_minimal.templates import (
    DEMONSTRATION,
    DEMONSTRATION_TEMPLATE,
    SYSTEM_TEMPLATE,
    COMMAND_TUTORIAL,
    INSTRUCTION_TEMPLATE,
)


def was_overflow_error(e: UnprocessableEntityError):
    response = e.response.json()
    return (
        "Input validation error: `inputs` tokens + `max_new_tokens` must be <="
        in response["error"]["message"]
    )


async def get_message(state: TaskState):
    overflowed = False
    output = None

    try:
        output = await get_model().generate(
            state.messages, state.tools, state.tool_choice
        )
        if output.stop_reason == "model_length":
            overflowed = True
    except UnprocessableEntityError as e:
        if was_overflow_error(e):
            overflowed = True
        else:
            raise e
    except BadRequestError as e:
        # convert each message to json and print it to a file
        with open(os.path.expanduser("~/messages.json"), "w") as f:
            for i, message in enumerate(state.messages):
                f.write(f"Message {i}:\n")
                f.write(str(message.model_dump_json()) + "\n\n")
        raise e

    return overflowed, output


class Agent:
    def __init__(
        self,
        native_function_calling: bool,
        wrong_command_limit: int,
        message_limit: int,
        task_instructions: str,
        state: TaskState,
    ):
        self.non_native_command_handler = CommandHandler(wrong_command_limit)
        self.state = state
        self.task_instructions = task_instructions
        self.state.message_limit = message_limit
        self.native_function_calling = native_function_calling

    def add_commands(self, commands: list[BaseCommand]):
        if self.native_function_calling:
            # provide equivalent tools for the commands
            self.state.tools.extend(command.native_tool() for command in commands)
            # force the model to use a tool
            self.state.tool_choice = "any"
        else:
            self.non_native_command_handler.add_commands(commands)

    async def loop(self):
        while not self.state.completed:
            overflowed, output = await get_message(self.state)

            if overflowed or output is None:
                self._shorten_message_history()
                continue

            if self.native_function_calling:
                await self._handle_native_function_call(output)
            else:
                await self._handle_non_native_function_calls(output)

    async def _handle_native_function_call(self, output: ModelOutput):
        self.state.output = output
        self.state.messages.append(output.message)
        tool_messages = await call_tools(output.message, self.state.tools)
        self.state.messages.extend(tool_messages)

    async def _handle_non_native_function_calls(self, output: ModelOutput):
        tool_output = await self.non_native_command_handler.get_command_output(output.completion)
        self.state.messages.append(output.message)
        self.state.messages.append(ChatMessageUser(content=tool_output))
        self.state.output = output


    def add_system_message(self):
        system_message = SYSTEM_TEMPLATE

        # we only add command tutorial and demonstration if the agent doesn't have native function calling
        if not self.native_function_calling:
            command_descriptions = "\n".join(
                [str(command) for command in self.non_native_command_handler.commands]
            )
            system_message += COMMAND_TUTORIAL.format(
                command_descriptions=command_descriptions
            )
            system_message += DEMONSTRATION_TEMPLATE.format(demonstration=DEMONSTRATION)

        if self.task_instructions != "":
            system_message += INSTRUCTION_TEMPLATE.format(
                instructions=self.task_instructions
            )

        append_system_message(
            self.state.messages, ChatMessageSystem(content=system_message)
        )

    def _shorten_message_history(self) -> None:
        """
        Removes the third message from the message history to handle overflow errors.
        Index 0 contains system message, index 1 contains task description,
        so we remove index 2.
        """
        msg2remove = self.state.messages.pop(2)
        if isinstance(msg2remove, ChatMessageAssistant):
            call_id_to_remove_set = set()
            for tool_call in msg2remove.tool_calls or []:
                call_id_to_remove_set.add(tool_call.id)
            self.state.messages = [
                msg
                for msg in self.state.messages
                if not isinstance(msg, ChatMessageTool)
                or msg.tool_call_id not in call_id_to_remove_set
            ]

    def submit_callback(self, answer: str, tool_result: str):
        self.state.completed = True
        self.state.output.completion = answer
