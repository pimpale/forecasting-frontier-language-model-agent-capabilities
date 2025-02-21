from collections import defaultdict
from pathlib import Path
from typing import Literal
from uuid import uuid4

from inspect_ai.util import SandboxEnvironment

Command = Literal[
    "view",
    "create",
    "insert",
    "string_replace",
    "undo_edit",
]

SNIPPET_LINES: int = 4
MAX_RESPONSE_LEN: int = 16_000


class EditTool:
    """File system editor tool for viewing, creating and editing files."""

    def __init__(self):
        self._file_history: dict[Path, list[str | None]] = defaultdict(list)

    async def execute(
        self,
        *,
        sandbox_env: SandboxEnvironment,
        command: Command,
        path: str,
        file_text: str | None = None,
        insert_line: int | None = None,
        new_str: str | None = None,
        old_str: str | None = None,
        view_range: list[int] | None = None,
    ) -> str:
        path_object = await self._validate_path(sandbox_env, command, path)

        if command == "view":
            await self._validate_view_args(view_range)
            return await self.view(sandbox_env, path_object, view_range)
        elif command == "create":
            await self._validate_create_args(file_text)
            return await self.create(sandbox_env, path_object, file_text)
        elif command == "string_replace":
            await self._validate_string_replace_args(old_str)
            return await self.string_replace(sandbox_env, path_object, old_str, new_str)
        elif command == "insert":
            await self._validate_insert_args(insert_line, new_str)
            return await self.insert(sandbox_env, path_object, insert_line, new_str)
        elif command == "undo_edit":
            return await self.undo_edit(sandbox_env, path_object)
        else:
            raise ValueError(f"Invalid command: {command}. Valid commands are: view, create, insert, string_replace.")

    async def _validate_view_args(self, view_range: list[int] | None) -> None:
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                raise ValueError(f"Invalid `view_range` parameter: {view_range}. It should be a list of two integers.")

            if view_range[1] != -1 and view_range[1] < view_range[0]:
                raise ValueError(
                    f"Invalid `view_range` parameter: {view_range}. Its second element `{view_range[1]}` should be "
                    f"larger or equal than its first `{view_range[0]}`."
                )

    async def _validate_create_args(self, file_text: str | None) -> None:
        if file_text is None:
            raise ValueError("Parameter `file_text` is required for command: `create`.")

    async def _validate_string_replace_args(self, old_str: str | None) -> None:
        if old_str is None:
            raise ValueError("Parameter `old_str` is required for command: `string_replace`.")

    async def _validate_insert_args(self, insert_line: int | None, new_str: str | None) -> None:
        if insert_line is None:
            raise ValueError("Parameter `insert_line` is required for command: `insert`.")
        if new_str is None:
            raise ValueError("Parameter `new_str` is required for command: `insert`.")

        if insert_line < 0:
            raise ValueError(f"Invalid `insert_line` parameter: {insert_line}. It should be non-negative.")

    async def _validate_path(self, sandbox_env: SandboxEnvironment, command: str, path: str) -> Path:
        if not isinstance(path, str):
            raise ValueError(f"Invalid `path` parameter: {path}. It should be a string.")
        path_object = Path(path)
        if not path_object.is_absolute():
            result = await sandbox_env.exec(["pwd"])
            current_dir = result.stdout.strip()
            suggested_path = Path(current_dir) / path
            raise ValueError(
                f"The path {path} is not an absolute path, it should start with `/`. Maybe you meant {suggested_path}?"
            )

        exists, error = await self._validate_path_exists(sandbox_env, path)
        if not exists and command != "create":
            raise ValueError(f"Path {path} does not exist: {error}. Please provide a valid path.")
        if exists and command == "create":
            raise ValueError(f"File already exists at {path}. Cannot overwrite with command `create`.")

        is_directory = await self._validate_is_directory(sandbox_env, path)
        if is_directory and command != "view":
            raise ValueError(f"The path {path} is a directory and only the `view` command can be used on directories.")

        return path_object

    async def _validate_path_exists(self, sandbox_env: SandboxEnvironment, path: Path) -> tuple[bool, str]:
        exists_result = await sandbox_env.exec(["ls", str(path)])
        return exists_result.returncode == 0, exists_result.stderr

    async def _validate_is_directory(self, sandbox_env: SandboxEnvironment, path: Path) -> tuple[bool, str]:
        is_dir_result = await sandbox_env.exec(["test", "-d", str(path)])
        is_directory = is_dir_result.returncode == 0 and not is_dir_result.stderr.strip()
        return is_directory

    async def view(self, sandbox_env: SandboxEnvironment, path: Path, view_range: list[int] | None = None) -> str:
        is_directory = await self._validate_is_directory(sandbox_env, path)
        if is_directory:
            return await self.view_directory(sandbox_env, path, view_range)
        else:
            return await self.view_file(sandbox_env, path, view_range)

    async def view_directory(
        self, sandbox_env: SandboxEnvironment, path: Path, view_range: list[int] | None = None
    ) -> str:
        if view_range:
            raise ValueError("The `view_range` parameter is not allowed when `path` points to a directory.")

        result = await sandbox_env.exec(["find", str(path), "-maxdepth", "2", "-not", "-path", "*/\\.*"])

        if result.stderr:
            stderr = await maybe_truncate(result.stderr, sandbox_env)
            return stderr

        stdout = await maybe_truncate(result.stdout, sandbox_env)
        return f"Files and directories up to 2 levels deep in {path} (excluding hidden):\n{stdout}\n"

    async def view_file(self, sandbox_env: SandboxEnvironment, path: Path, view_range: list[int] | None = None) -> str:
        file_content = await self._read_file(sandbox_env, path)
        init_line = 1

        if view_range:
            file_lines = file_content.split("\n")
            n_lines = len(file_lines)
            init_line, final_line = view_range

            if init_line < 1 or init_line > n_lines:
                raise ValueError(
                    f"Invalid `view_range` parameter: {view_range}. Its first element `{init_line}` should be "
                    f"within the range of lines of the file: {[1, n_lines]}."
                )

            if final_line > n_lines:
                raise ValueError(
                    f"Invalid `view_range` parameter: {view_range}. Its second element `{final_line}` should be "
                    f"smaller than the number of lines in the file: `{n_lines}`."
                )

            if final_line != -1 and final_line < init_line:
                raise ValueError(
                    f"Invalid `view_range` parameter: {view_range}. Its second element `{final_line}` should be "
                    f"larger or equal than its first `{init_line}`."
                )

            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])

        return await self._format_file_for_model(sandbox_env, file_content, str(path), init_line=init_line)

    async def create(self, sandbox_env: SandboxEnvironment, path: Path, file_text: str) -> str:
        await self._write_file(sandbox_env, path, file_text)
        self._file_history[path].append(None)
        return f"File created successfully at: {path}."

    async def _write_file(self, sandbox_env: SandboxEnvironment, path: Path, content: str) -> None:
        try:
            await sandbox_env.write_file(str(path), content)
        except Exception as err:
            raise ValueError(f"Error writing to {path}: {err}.") from err

    async def insert(self, sandbox_env: SandboxEnvironment, path: Path, insert_line: int, new_str: str) -> str:
        file_content = await self._read_file(sandbox_env, path)
        file_content = file_content.expandtabs()
        new_str = new_str.expandtabs()

        file_text_lines = file_content.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line > n_lines_file:
            raise ValueError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range "
                f"of lines of the file: {[0, n_lines_file]}."
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = file_text_lines[:insert_line] + new_str_lines + file_text_lines[insert_line:]
        # Create snippet exactly like Anthropic version
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        await self._write_file(sandbox_env, path, new_file_text)
        self._file_history[path].append(file_content)

        success_msg = f"The file {path} has been edited. "
        output = await self._format_file_for_model(
            sandbox_env,
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += output
        success_msg += (
            "Review the changes and make sure they are as expected (correct indentation, "
            "no duplicate lines, etc). Edit the file again if necessary."
        )

        return success_msg

    async def string_replace(
        self, sandbox_env: SandboxEnvironment, path: Path, old_str: str, new_str: str | None
    ) -> str:
        file_content = await self._read_file(sandbox_env, path)
        file_content = file_content.expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str else ""

        await self._validate_occurences_are_unique(path, file_content, old_str)
        new_content = await self.replace_string(sandbox_env, path, file_content, old_str, new_str)
        return await self._create_edit_snippet(sandbox_env, path, file_content, old_str, new_str, new_content)

    async def _validate_occurences_are_unique(self, path: Path, file_content: str, old_str: str) -> None:
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise ValueError(f"No replacement was performed, `{old_str}` did not appear verbatim in {path}.")
        elif occurrences > 1:
            lines = [idx + 1 for idx, line in enumerate(file_content.split("\n")) if old_str in line]
            raise ValueError(
                f"No replacement was performed. Multiple occurrences of `{old_str}` in lines {lines}. "
                "Please ensure it is unique."
            )

    async def replace_string(
        self, sandbox_env: SandboxEnvironment, path: Path, file_content: str, old_str: str, new_str: str
    ) -> str:
        new_content = file_content.replace(old_str, new_str)
        await self._write_file(sandbox_env, path, new_content)
        self._file_history[path].append(file_content)
        return new_content

    async def _create_edit_snippet(
        self,
        sandbox_env: SandboxEnvironment,
        path: Path,
        file_content: str,
        old_str: str,
        new_str: str,
        new_content: str,
    ) -> str:
        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_content.split("\n")[start_line : end_line + 1])

        msg = f"The file {path} has been edited. "
        output = await self._format_file_for_model(sandbox_env, snippet, f"a snippet of {path}", start_line + 1)
        msg += output
        msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."
        return msg

    async def _read_file(self, sandbox_env: SandboxEnvironment, path: Path) -> str:
        try:
            return await sandbox_env.read_file(str(path))
        except Exception as err:
            raise ValueError(f"Error reading {path}: {err}") from err

    async def undo_edit(self, sandbox_env: SandboxEnvironment, path: Path) -> str:
        if not self._file_history[path]:
            raise ValueError(f"No edit history for {path}.")

        old_content = self._file_history[path].pop()
        if not old_content:
            # If the last edit was a create, we need to delete the file
            # Note in Anthropic's implementation, they don't delete the file, they
            # keep the content to the content at creation time.
            await sandbox_env.exec(["rm", str(path)])
            msg = f"Undone creation of file {path}. The file has been deleted."
        else:
            await self._write_file(sandbox_env, path, old_content)
            msg = f"Last edit to {path} undone successfully. "
            output = await self._format_file_for_model(sandbox_env, old_content, str(path))
            msg += output
        return msg

    async def _format_file_for_model(
        self,
        sandbox_env: SandboxEnvironment,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ) -> str:
        """Format file content with line numbers."""
        file_content = await maybe_truncate(file_content, sandbox_env)

        if expand_tabs:
            file_content = file_content.expandtabs()

        # Add line numbers with proper formatting
        file_content = "\n".join([f"{i + init_line:6}\t{line}" for i, line in enumerate(file_content.split("\n"))])

        return f"Here's the result of running `cat -n` on {file_descriptor}:\n" + file_content + "\n"


async def save_long_output(content: str, sandbox_env: SandboxEnvironment) -> tuple[Path, str]:
    """Save long output to a file and return the path and truncated message."""
    filename = f"long_output_{uuid4().hex[:8]}.txt"
    filepath = Path("/tmp") / filename
    await sandbox_env.write_file(str(filepath), content)

    msg = (
        f"<response clipped> <NOTE>The output for the last command was too long to display. "
        f"The full output of the command was saved to '{filepath}'. You can search inside the file "
        f"with `grep -n` to find the line numbers of what you are looking for, e.g. "
        f"`grep -n 'search_string' {filepath}`. You should retry this viewing tool with the range parameter "
        f"to view what you are looking for."
    )
    return filepath, msg


async def maybe_truncate(
    content: str, sandbox_env: SandboxEnvironment, truncate_after: int | None = MAX_RESPONSE_LEN
) -> str:
    """Truncate content if it exceeds the specified length."""
    if not truncate_after or len(content) <= truncate_after:
        return content

    _, truncated_message = await save_long_output(content, sandbox_env)
    return content[:truncate_after] + truncated_message
