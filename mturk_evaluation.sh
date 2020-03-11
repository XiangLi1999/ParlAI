#!/bin/bash
source /home/aadelucia/miniconda3/bin/activate
conda activate dialogue

PROJECT_HOME=/home/aadelucia/files/course_projects/discourse-hw4/ParlAI

# Log into Heroku
# $PROJECT_HOME/parlai/mturk/tmp/heroku-cli-v6.99.0-ec9edad-linux-x64/bin/heroku login

python "$PROJECT_HOME/mturk_evaluation/run.py" --sandbox \
   -nc 4 \
   --model-file "$PROJECT_HOME/parlai_internal/forward.ckpt.checkpoint" \
	 --model-file-backward $PROJECT_HOME/parlai_internal/backward.ckpt.checkpoint \
	 --no-cuda \
	 --beam-size 8 \
	 --inference beam
