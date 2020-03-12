#!/bin/bash
source /home/aadelucia/miniconda3/bin/activate
conda activate dialogue

PROJECT_HOME=/home/aadelucia/files/course_projects/discourse-hw4/ParlAI

python -m parlai.scripts.interactive \
	-mf $PROJECT_HOME/parlai_internal/forward_finetune.ckpt.checkpoint \
	--model transformer/generator \
	--no-cuda \
	--beam-size 8 \
	--inference "beam"
