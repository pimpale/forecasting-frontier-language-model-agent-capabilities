#!/bin/bash

MODEL=gpt-4o-2024-08-06
LOGDIR=eval_set_logs/${MODEL//\//--}_subset_nonnative # replace / with --

inspect eval tasks/swebench_verified_subset.py \
	--model openai/$MODEL \
	--log-dir $LOGDIR \
	-T resume_dir=$(realpath $LOGDIR) \
	-T native_function_calling=False
