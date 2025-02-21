#!/bin/bash

MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
LOGDIR=eval_set_logs/${MODEL//\//--}

inspect eval tasks/swebench_verified_full.py \
	--model together/$MODEL \
	--log-dir $LOGDIR \
	-T resume_dir=$(realpath $LOGDIR) \
	-T native_function_calling=False