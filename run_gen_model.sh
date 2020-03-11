#!/bin/bash

python -m parlai.scripts.interactive \
	-mf parlai_internal/forward.ckpt.checkpoint \
	--model transformer/generator \
	--no-cuda \
	--beam-size 8 \
	--inference "beam"
