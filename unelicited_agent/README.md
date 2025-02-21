# Inspect Agent: Bash Minimal

## Environment Setup

First, install the bash_minimal agent package:
```bash
pip install -e .
```

Then, you'll need to create a `.env` file in the root directory with the following configuration:
```env
INSPECT_EVAL_MODEL=openai/gpt-4o
OPENAI_API_KEY=...
```

The `INSPECT_EVAL_MODEL`environment variable is required for specifying which model Inspect should call.

To run the task:
```bash
inspect eval task.py --trace
```