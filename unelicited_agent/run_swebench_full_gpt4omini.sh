#!/bin/bash

MODEL=gpt-4o-mini-2024-07-18
LOGDIR=eval_set_logs/$MODEL

inspect eval tasks/swebench_verified_full.py \
	--model openai/$MODEL \
	--log-dir $LOGDIR \
	-T resume_dir=$(realpath $LOGDIR)
