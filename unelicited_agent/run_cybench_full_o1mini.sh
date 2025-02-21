#!/bin/bash

MODEL=o1-mini-2024-09-12
LOGDIR=cybench_eval_set_logs/${MODEL//\//--}

inspect eval tasks/cybench_full.py \
	--model openai/$MODEL \
	--log-dir $LOGDIR \
	-T resume_dir=$(realpath $LOGDIR)