#!/bin/bash

MODEL=gpt-3.5-turbo-0125
LOGDIR=cybench_eval_set_logs/${MODEL//\//--}

inspect eval tasks/cybench_full.py \
	--model openai/$MODEL \
	--log-dir $LOGDIR \
	-T resume_dir=$(realpath $LOGDIR)