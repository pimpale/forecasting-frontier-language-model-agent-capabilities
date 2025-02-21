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
                input="YOUR TASK: Calculate 2 to the power of 100 with python, and submit the answer.",
                target="1267650600228229401496703205376",
            )
        ],
        solver=solver,
        scorer=includes(),
        sandbox="docker",
    )
