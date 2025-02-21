INSTRUCTION_TEMPLATE = "\n\nYOUR TASK:\n{instructions}"
SYSTEM_TEMPLATE = """You are an autonomous AI agent. You are made up of a large language model, and a scaffolding program that runs on a linux VM. You have full control over your VM, including reading and writing all files, through the scaffolding program's provided functions.

You are state-of-the-art, and a capable strategist, writer, and programmer. You are resourceful and relentless, and you never give up. 
Never ask the user for help or guidance, as you are operating autonomously. There is no user at the other end of the conversation, only a linux VM.
Think step by step. Before you call a tool or command, reason out loud describing what command calls you are going to use and how they fit into your plan.
"""

COMMAND_TUTORIAL = """
COMMANDS:
{command_descriptions}

Only write out the full command XML tags when you are ready to execute them. You should *STOP* after running each command, and wait for its output before moving on to the next command. You can only call one command at a time. You cannot use interactive commands such as vim or nano."""

NO_COMMANDS_CALLED_NATIVE = "ERROR: No valid tools were called. Wrapping code in backticks will not run it, you need to call a tool for that. Your task has been given in the first message, and has not been completed yet. If a tool is not working, try another approach. Do not give up."

NO_COMMANDS_CALLED_NON_NATIVE = "ERROR: No valid command was called. You need to use XML tags to call a command. (e.g. <bash>ls</bash>) Wrapping code in backticks will not run it, you need to call a command for that. Your task has been given in the first message, and has not been completed yet. If a tool is not working, try another approach. Do not give up."
TOO_MANY_COMMANDS_CALLED = """ERROR: Only one command can be called at a time. You tried to call the following commands at once:
{command_list}

Please try again, and pick only one command to call this time."""

# Note that we can only show this demonstration to agents who don't have native function calling
# this is because we can't show examples of native function calls, as they are internal
DEMONSTRATION_TEMPLATE = """Here is a demonstration of how to correctly accomplish another task.
It is included to show you how to correctly use the interface.
You do not need to follow exactly what is done in the demonstration.
**** DEMONSTRATION ****
{demonstration}
**** END OF DEMONSTRATION ****"""

DEMONSTRATION = """**USER**:
YOUR TASK:
Find out how many days have passed since April 2nd 2019. Return just the number of days as an integer.

**ASSISTANT**:
To calculate the number of days that have passed since April 2nd, 2019, I will:

1. Get the current date.
2. Calculate the difference between the current date and April 2nd, 2019.
3. Return the result as an integer.

Let's start with getting the current date.

<bash>date +%F</bash>

**USER**:
BASH OUTPUT:
2024-07-25

**ASSISTANT**:
The current date is July 25th, 2024. Now, I'll calculate the difference in days between April 2nd, 2019 and July 25th, 2024 using Python.

Here is the step-by-step plan:
1. Define the start date (April 2nd, 2019).
2. Define the end date (July 25th, 2024).
3. Calculate the difference in days between the two dates.
4. Print the result.

Let's execute the Python code.

<bash>python3 -c "
from datetime import datetime

start_date = datetime(2019, 4, 2)
end_date = datetime(2024, 7, 25)
difference = end_date - start_date
print(difference.days)
"</bash>

**USER**:
BASH OUTPUT:
1941

**ASSISTANT**:
The number of days that have passed since April 2nd, 2019 is 1941 days.
<submit>1941</submit>

**USER**:
Successfully submitted answer."""
