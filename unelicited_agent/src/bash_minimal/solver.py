from dataclasses import dataclass

from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import sandbox

from bash_minimal.Agent import Agent
from bash_minimal.commands.BashCommand import BashCommand
from bash_minimal.commands.EditCommand import EditCommand
from bash_minimal.commands.SubmitCommand import SubmitCommand


@dataclass(frozen=True)
class AgentArguments:
    native_function_calling: bool
    task_instructions: str
    wrong_command_limit: int = (
        8  # The maximum number of messages with wrong commands in a row.
    )
    bash_timeout: int = 600  # The maximum seconds a bash command can run for.
    max_output_length: int = (
        1500  # The maximum characters in the output of a bash command.
    )
    message_limit: int = 50  # The maximum number of assistant messages.


def initialize_agent(args: AgentArguments, state: TaskState):
    agent = Agent(
        args.native_function_calling,
        args.wrong_command_limit,
        args.message_limit,
        args.task_instructions,
        state,
    )
    commands = [
        BashCommand(args.bash_timeout, args.max_output_length),
        EditCommand(),
        SubmitCommand(agent.submit_callback),
    ]
    agent.add_commands(commands)
    agent.add_system_message()
    return agent


@solver
def bash_minimal(native_function_calling: bool = True, task_instructions: str = ""):
    async def solve(state: TaskState, generate: Generate):
        args = AgentArguments(native_function_calling, task_instructions)
        agent = initialize_agent(args, state)
        await agent.loop()
        return state

    return solve
