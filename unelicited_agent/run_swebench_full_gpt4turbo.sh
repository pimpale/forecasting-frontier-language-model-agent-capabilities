#!/bin/bash

MODEL=gpt-4-turbo-2024-04-09
LOGDIR=eval_set_logs/$MODEL

inspect eval tasks/swebench_verified_full.py \
	--model openai/$MODEL \
	--log-dir $LOGDIR \
	-T resume_dir=$(realpath $LOGDIR)
