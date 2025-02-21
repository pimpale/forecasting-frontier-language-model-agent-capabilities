"""File system editor tool that allows viewing, creating, and editing files."""

from typing import Literal, cast, get_args

from inspect_ai.tool import Tool, tool
from inspect_ai.tool._tool_call import ToolCall, ToolCallContent, ToolCallView, ToolCallViewer
from inspect_ai.util import sandbox, store

from bash_minimal.commands.edit_tool.store import store_contains, store_load, store_overwrite_existing, store_save
from bash_minimal.commands.edit_tool.file_editing import Command, EditTool


def edit_viewer(tool_name: str) -> ToolCallViewer:
    """Creates a viewer for edit tool and formats the output as markdown."""

    def viewer(tool_call: ToolCall) -> ToolCallView:
        cmd = tool_call.arguments.get("cmd", tool_call.function).strip()

        content = []

        args = []
        for key, value in tool_call.arguments.items():
            if value is not None and key != "cmd":
                if isinstance(value, str) and "\n" in value:
                    args.append(f"- **{key}**:\n```\n{value}\n```")
                else:
                    args.append(f"- **{key}**: `{value}`")

        if args:
            content.append("**Args:**")
            content.extend(args)

        return ToolCallView(
            call=ToolCallContent(title=f"{tool_name} {cmd}", format="markdown", content="\n".join(content))
        )

    return viewer


def get_edit_tool() -> EditTool:
    """Get or create an EditTool instance for the current sample."""
    if not store_contains(EditTool, store()):
        edit_tool = EditTool()
        store_save(edit_tool, store())
    else:
        edit_tool = store_load(EditTool, store())

    return edit_tool


def validate_command(cmd: str) -> Command:
    if cmd not in get_args(Command):
        raise ValueError(f"Invalid command '{cmd}'. Allowed commands are: {', '.join(get_args(Command))}")
    return cast(Command, cmd)


# Note: Be aware of parallel tool calls! Currently this is allowed, and this might have unexpected side effects
# e.g. if you try to create a file and undo the edit at the same time, the tool might first
# execute the undo and then the create, which will fail.
# We do want to allow for parallel calls however, as long as they are not editing the same file.
@tool(viewer=edit_viewer("edit_tool"))
def edit_tool(sandbox_name: str = "default") -> Tool:
    """Create an edit tool instance.

    The edit tool provides file system operations like viewing, creating and editing files
    in a safe sandbox environment.

    Args:
        sandbox_env: The sandbox environment to operate in

    Returns:
        Tool: An async function that executes edit commands
    """
    # The following docstring is inspired by anthropic's tool description (see https://www.anthropic.com/research/swe-bench-sonnet)
    EXECUTE_DOCSTRING = """Custom editing tool for viewing, creating and editing files.

        * State persists across command calls and user discussions
        * If `path` is a file, `view` shows `cat -n` output. If `path` is a directory, `view` lists \
non-hidden files 2 levels deep
        * The `create` command fails if `path` already exists as a file
        * Long outputs will be truncated and marked with `<response clipped>` 
        * The `undo_edit` command reverts the last edit made to a file at `path`. If the previous command was a\
`create`, the file is deleted.

        Notes for using the `string_replace` command:
        * The `old_str` parameter must match exactly one or more consecutive lines from the original file. \
Be mindful of whitespaces!
        * If `old_str` isn't unique in the file, no replacement occurs. Include enough context in `old_str` \
for uniqueness
        * The `new_str` parameter should contain the lines that replace `old_str`

        Args:
            cmd (str): The commands to run. Allowed options are: `view`, `create`, `string_replace`, `insert`, \
`undo_edit`.
            path (str): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo` or `/file.py`.
            file_text (str | None): Required Parameter of `create` command, with the content of the file to be created.
            insert_line (int | None): Required Parameter of `insert` command. `new_str` is inserted AFTER the line \
`insert_line` of `path`.
            new_str (str | None): Required Parameter of `string_replace` command containing the new string. \
Required Parameter of `insert` command containing the string to insert.
            old_str (str | None): Required Parameter of `string_replace` command containing the string in `path` \
to replace.
            view_range (list[int] | None): Required Parameter of `view` command when `path` points to a file. \
If empty list is given, the full file is shown. If provided, the file will be shown in the \
indicated line number range, e.g. [11, 12] will show lines 11 and 12. \
Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end \
of the file.

        Returns:
            str: Result of the executed command
        """

    async def execute(
        cmd: Literal[
            "view",
            "create",
            "insert",
            "string_replace",
            "undo_edit",
        ],
        path: str,
        file_text: str | None = "",
        insert_line: int | None = -1,
        new_str: str | None = "",
        old_str: str | None = "",
        view_range: list[int] | None = [],  # noqa: B006
    ) -> str:
        try:
            command = validate_command(cmd)

            # Convert empty defaults back to None for internal API
            # If inspect ever allows to provide None as default value, we can remove this
            file_text_arg = file_text if file_text != "" else None
            insert_line_arg = insert_line if insert_line != -1 else None
            new_str_arg = new_str if new_str != "" else None
            old_str_arg = old_str if old_str != "" else None
            view_range_arg = view_range if view_range != [] else None

            sandbox_env = sandbox(sandbox_name)
            edit_tool = get_edit_tool()

            result = await edit_tool.execute(
                sandbox_env=sandbox_env,
                command=command,
                path=path,
                file_text=file_text_arg,
                insert_line=insert_line_arg,
                new_str=new_str_arg,
                old_str=old_str_arg,
                view_range=view_range_arg,
            )
            # Since EditTool modifies its internal state (_file_history dictionary)
            # we need to explicitly save back any modifications
            store_overwrite_existing(edit_tool, store())

            return result

        except ValueError as e:
            return f"ValueError: {str(e)}"
        except Exception as e:
            return f"An error occurred: {str(e)}"

    execute.__doc__ = EXECUTE_DOCSTRING

    return execute
