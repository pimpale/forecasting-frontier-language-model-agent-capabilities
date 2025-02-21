#!/bin/bash

MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
LOGDIR=cybench_eval_set_logs/${MODEL//\//--}

inspect eval tasks/cybench_full.py \
	--model together/$MODEL \
	--log-dir $LOGDIR \
	-T resume_dir=$(realpath $LOGDIR) \
	-T native_function_calling=False