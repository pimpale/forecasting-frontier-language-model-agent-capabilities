#!/bin/bash

MODEL=gpt-4o-2024-08-06
LOGDIR=cybench_eval_set_logs/${MODEL//\//--}

inspect eval tasks/cybench_full.py \
	--model openai/$MODEL \
	--log-dir $LOGDIR \
	-T resume_dir=$(realpath $LOGDIR)