from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_evals.cybench import cybench

from bash_minimal.solver import bash_minimal


@task
def example_task(resume_dir: str, native_function_calling=True) -> Task:
    assert resume_dir is not None and Path(resume_dir).is_absolute()
    solver = bash_minimal(
        native_function_calling=native_function_calling,
    )
    
    task = cybench(
        agent=solver,
        max_messages=50,
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
    
    # generous token limit of 1 mil tokens
    task.token_limit = 1_000_000

    return task
