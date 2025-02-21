#!/bin/bash

MODEL=Qwen/Qwen2.5-72B-Instruct-Turbo
LOGDIR=eval_set_logs/${MODEL//\//--} # replace / with --

inspect eval tasks/swebench_verified_subset.py \
	--model together/$MODEL \
	--log-dir $LOGDIR \
	-T resume_dir=$(realpath $LOGDIR) \
	-T native_function_calling=False