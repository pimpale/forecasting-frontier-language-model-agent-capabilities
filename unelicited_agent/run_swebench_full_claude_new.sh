#!/bin/bash

MODEL=claude-3-5-sonnet-20241022
LOGDIR=eval_set_logs/$MODEL

inspect eval tasks/swebench_verified_full.py \
	--model anthropic/$MODEL \
	--log-dir $LOGDIR \
	-T resume_dir=$(realpath $LOGDIR)
