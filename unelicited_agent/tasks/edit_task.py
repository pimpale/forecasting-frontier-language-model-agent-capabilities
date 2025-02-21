from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes

from bash_minimal.solver import bash_minimal


@task
def example_task() -> Task:
    solver = bash_minimal()

    return Task(
        dataset=[
            Sample(
                input="YOUR TASK: Create a file called 'file.txt' and write 'Hello, World!' to it. Once done, submit the phrase 'Hello, World!'.",
                target="Hello, World!",
            )
        ],
        solver=solver,
        scorer=includes(),
        sandbox="docker",
    )
