#!/bin/bash
source /home/aadelucia/miniconda3/bin/activate
conda activate dialogue

PROJECT_HOME=/home/aadelucia/files/course_projects/discourse-hw4/ParlAI

python -m parlai.scripts.interactive \
  --model transformer/generatorMMI \
  --model-file $PROJECT_HOME/parlai_internal/forward_finetune.ckpt \
  --no-cuda \
  --beam-size 8 \
  --inference beam \
  --model-file-backward $PROJECT_HOME/parlai_internal/backward.ckpt.checkpoint \
