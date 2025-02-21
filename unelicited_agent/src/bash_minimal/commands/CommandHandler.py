from bash_minimal.commands.BaseCommand import BaseCommand
from bash_minimal.templates import NO_COMMANDS_CALLED_NON_NATIVE, TOO_MANY_COMMANDS_CALLED


class CommandHandler:
    def __init__(self, wrong_command_limit: int):
        self.commands: list[BaseCommand] = []
        self.wrong_command_limit = wrong_command_limit
        self.wrong_command_in_a_row = 0

    def too_many_wrong_commands(self) -> bool:
        return self.wrong_command_in_a_row >= self.wrong_command_limit

    async def get_command_output(self, content: str) -> str:
        total_call_num = self._total_command_calls(content)
        command_output = None
        wrong_command_call = False

        if total_call_num == 0:
            wrong_command_call = True
            command_output = NO_COMMANDS_CALLED_NON_NATIVE
        elif total_call_num > 1:
            wrong_command_call = True
            command_output = self.handle_too_many_commands(content)
        else:
            command_output, wrong_command_call = await self._execute_command(content)

        if wrong_command_call:
            self.wrong_command_in_a_row += 1
        else:
            self.wrong_command_in_a_row = 0

        return command_output

    def _total_command_calls(self, content: str) -> int:
        return sum(command.count_occurrences(content) for command in self.commands)

    def handle_too_many_commands(self, content: str) -> str:
        tags = self._get_all_tags(content)
        command_list = "\n".join(f"{i}. {tag}" for i, tag in enumerate(tags, 1))

        return TOO_MANY_COMMANDS_CALLED.format(command_list=command_list)

    def _get_all_tags(self, content: str) -> list[str]:
        tags = []
        for command in self.commands:
            tags.extend(command.extract_full_tags(content))
        return tags

    async def _execute_command(self, content: str) -> tuple[str, bool]:
        for command in self.commands:
            outputs = await command.execute(content)

            for output in outputs:
                if hasattr(output, "failed") and output.failed:
                    return output.content, True
                else:
                    return output.content, False

        raise ValueError("No command output found")

    def add_commands(self, commands: list[BaseCommand]):
        self.commands.extend(commands)
