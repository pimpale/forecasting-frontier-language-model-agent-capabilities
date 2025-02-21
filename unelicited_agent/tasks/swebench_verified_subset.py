from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.scorer import includes

from inspect_evals.swe_bench import swe_bench
from inspect_evals.swe_bench import swe_bench_baseline_scorer, swe_bench_scorer

from bash_minimal.solver import bash_minimal


@task
def example_task(resume_dir: str, native_function_calling=True) -> Task:
    assert resume_dir is not None and Path(resume_dir).is_absolute()
    
    solver = bash_minimal(
        native_function_calling=native_function_calling,
        task_instructions="""
Below is a Github Issue that describes the problem you need to solve.
The repository has been cloned into /testbed (your current working directory). All dependencies are installed for you.
Your job is as follows:
1. Isolate the file/files where the bug is found.
2. Fix the bug.
3. Write a new test case that demonstrates the bug. The test case should be written in the same style as the existing tests.
4. Run the tests to ensure that the bug is fixed. Please ensure that you follow proper instructions to run tests, as described in the repo's documentation. Not all repositories have the same test running instructions. Note that some tests may fail, but as long as they are not related to the bug you fixed, you can ignore them.
5. Submit the empty string using the submit tool.

ALWAYS EXPLAIN YOUR REASONING BEFORE RUNNING A COMMAND. This will help you avoid mistakes and make it easier for us to understand your thought process.
"""
    )

    task = swe_bench(
        dataset="princeton-nlp/SWE-bench_Verified",
        instance_ids=[
            "django__django-10097",
            "django__django-12155",
            "django__django-14855",
            "django__django-7530",
            "pydata__xarray-7393",
            "pytest-dev__pytest-8399",
            "sympy__sympy-15809",
        ],
        solver=solver,
        scorer=swe_bench_scorer(),
    )
    
    # Filter out samples that have already been processed
    already_processed_ids = set()
    for log_file in list_eval_logs(resume_dir):
        print(f"Reading log file: {log_file}")
        log = read_eval_log(log_file)
        for sample in log.samples or []:
            if sample.scores is not None:
                already_processed_ids.add(sample.id)

    
    task.dataset = task.dataset.filter(lambda x: x.id not in already_processed_ids)
    
    
    for sample in task.dataset:
        with open(f"samples/{sample.id}.json", "w") as f:
            f.write(sample.model_dump_json())
    
    return task