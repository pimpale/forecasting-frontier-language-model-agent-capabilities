#!/bin/bash

MODEL=gpt-4-turbo-2024-04-09
LOGDIR=cybench_eval_set_logs/${MODEL//\//--}

inspect eval tasks/cybench_full.py \
	--model openai/$MODEL \
	--log-dir $LOGDIR \
	-T resume_dir=$(realpath $LOGDIR)