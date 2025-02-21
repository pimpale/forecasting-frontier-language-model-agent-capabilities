from collections import defaultdict
from typing import Any, Optional, override

from inspect_ai.tool import Tool
from inspect_ai.util import sandbox

from bash_minimal.commands.BaseCommand import (
    BaseCommand,
    CallbackType,
    CommandOutput,
)
from bash_minimal.commands.edit_tool.file_editing import EditTool
from bash_minimal.commands.edit_tool.file_edit_tool import edit_tool

import xml.parsers.expat

class EditCommand(BaseCommand):
    def __init__(self, callback: Optional[CallbackType] = None):
        super().__init__(
            xml_tag="edit",
            description=
"""A tool for viewing, creating and editing files.
        * State persists across command calls and user discussions
        * If `path` is a file, `view` shows `cat -n` output. If `path` is a directory, `view` lists \
non-hidden files 2 levels deep
        * The `create` command fails if `path` already exists as a file
        * Long outputs will be truncated and marked with `<response clipped>` 
        * The `undo_edit` command reverts the last edit made to a file at `path`
        * For non-necessary parameters use "" (strings), [] (lists), or -1 (integers)

        Notes for using the `string_replace` command:
        * The `old_str` parameter must match exactly one or more consecutive lines from the original file. \
Be mindful of whitespaces and indentation!
        * If `old_str` isn't unique in the file, no replacement occurs. Include enough context in `old_str` \
for uniqueness
        * The `new_str` parameter should contain the lines that replace `old_str`

        Args:
            cmd (str): The commands to run. Allowed options are: `view`, `create`, `string_replace`, `insert`, \
`undo_edit`.
            path (str): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo` or `/file.py`.
            file_text (str): Required Parameter of `create` command, with the content of the file to be created.
            insert_line (int): Required Parameter of `insert` command. `new_str` is inserted AFTER the line \
`insert_line` of `path`.
            new_str (str): Required Parameter of `string_replace` command containing the new string. \
Required Parameter of `insert` command containing the string to insert. Please encode angle brackets as \
`&lt;` and `&gt;` for `<` and `>` respectively.
            old_str (str): Required Parameter of `string_replace` command containing the string in `path` \
to replace. Please encode angle brackets as `&lt;` and `&gt;` for `<` and `>` respectively.
            view_range (list[int]): Required Parameter of `view` command when `path` points to a file. \
If empty list is given, the full file is shown. If provided, the file will be shown in the \
indicated line number range, e.g. [11, 12] will show lines 11 and 12. \
Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end \
of the file.

        Returns:
            str: Result of the executed command
            
Examples:
To view a specific range of lines in a file:
<edit>
<cmd>view</cmd>
<path>/repo/some_other_file.py</path>
<view_range>[2, 4]</view_range>
</edit>
To create a file:
<edit>
<cmd>create</cmd>
<path>/repo/another_file.py</path>
<file_text>print('Hello, World!')</file_text>
</edit>
To replace a string in a file:
<edit>
<cmd>string_replace</cmd>
<path>/repo/random_file.py</path>
<old_str>    print('Hello, World!')</old_str>
<new_str>    if x == 1:
        print('Hello, World!')
    else:
        print('Goodbye, World!')</new_str>
</edit>
To insert a string after a specific line in a file:
<edit>
<cmd>insert</cmd>
<path>/repo/another_file.py</path>
<insert_line>2</insert_line>
<new_str>print('Goodbye, World!')</new_str>
</edit>
To undo the last edit made to a file:
<edit>
<cmd>undo_edit</cmd>
<path>/repo/yet_another_file.py</path>
</edit>
""",
            callback=callback,
        )
        self.edit_tool = EditTool()


    @override
    def native_tool(self) -> Tool:
        return edit_tool()

    async def _run(self, content: str) -> CommandOutput:
        content = "<edit>" + content + "</edit>"

        # problem: we are presented with a set of xml tags  (whatever is contained within <edit> </edit>)
        # We need to split the content into the individual tags
        # For each tag, we encode into a dictionary
        # finally, we pass it into the edit_tool to execute the command
     
        d: defaultdict[str, Any] = defaultdict(lambda: None)
        element_stack: list[str] = []
    
        def end_element(name: str):
            element_stack.pop()
    
        def start_element(name:str, attrs):
            element_stack.append(name)
        
        def char_data(data):
            data = data.replace("&lt;", "<").replace("&gt;", ">")
            if d[element_stack[-1]] is None:
                d[element_stack[-1]] = data
            else:
                d[element_stack[-1]] += data

        p = xml.parsers.expat.ParserCreate()
        p.StartElementHandler = start_element
        p.EndElementHandler = end_element
        p.CharacterDataHandler = char_data
        
        try:
            p.Parse(content, True)
        except Exception as e:
            return CommandOutput(str(e))

        # parse view range (if it exists)
        if d["view_range"]:
            nums = [i for i in d["view_range"].strip("[]").split(",")]                
            if nums == [""]:
                d["view_range"] = []
            else:
                for n in nums:
                    if not n.strip().isdigit():
                        return CommandOutput("ERROR: view_range must contain only positive integers")
                d["view_range"] = [int(i) for i in nums]

        # parse insert line (if it exists)
        if d["insert_line"]:
            d["insert_line"] = int(d["insert_line"])

        try:
            resp = await self.edit_tool.execute(
                sandbox_env=sandbox(),
                command=d["cmd"],
                path=d["path"],
                file_text=d["file_text"],
                view_range=d["view_range"],
                old_str=d["old_str"],
                new_str=d["new_str"],
                insert_line=d["insert_line"],
            )
        except Exception as e:
            resp = str(e)
        
        return CommandOutput(resp)