#!/bin/bash

MODEL=claude-2.1
LOGDIR=eval_set_logs/$MODEL

inspect eval tasks/swebench_verified_full.py \
	--model anthropic/$MODEL \
	--log-dir $LOGDIR \
	-T resume_dir=$(realpath $LOGDIR) \
	-T native_function_calling=False
