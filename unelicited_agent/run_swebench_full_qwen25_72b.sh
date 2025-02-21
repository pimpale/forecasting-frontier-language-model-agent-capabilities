#!/bin/bash

MODEL=Qwen/Qwen2.5-72B-Instruct-Turbo
LOGDIR=eval_set_logs/${MODEL//\//--}

inspect eval tasks/swebench_verified_full.py \
	--model together/$MODEL \
	--log-dir $LOGDIR \
	--max-connections 3 \
	-T resume_dir=$(realpath $LOGDIR) \
	-T native_function_calling=False