#!/bin/bash

MODEL=claude-3-5-sonnet-20241022
LOGDIR=eval_set_logs/${MODEL//\//--}_subset_nonnative # replace / with --

inspect eval tasks/swebench_verified_subset.py \
	--model anthropic/$MODEL \
	--log-dir $LOGDIR \
	-T resume_dir=$(realpath $LOGDIR) \
	-T native_function_calling=False
