#!/bin/bash

MODEL=o1-2024-12-17
LOGDIR=eval_set_logs/$MODEL

inspect eval tasks/swebench_verified_full.py \
	--model openai/$MODEL \
	--log-dir $LOGDIR \
	--max-connections 3 \
	-T resume_dir=$(realpath $LOGDIR)
