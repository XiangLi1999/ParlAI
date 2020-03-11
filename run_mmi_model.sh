#!/bin/bash

python -m parlai.scripts.interactive \
	-mf parlai_internal/forward.ckpt.checkpoint \
	--model-file-backward parlai_internal/backward.ckpt.checkpoint \
	--model transformer/generatorMMI \
	--no-cuda \
	--beam-size 8 \
	--inference "beam"
