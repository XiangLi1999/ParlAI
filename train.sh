#!/bin/bash

python -u examples/train_model.py \
	-t opensubtitles \
	-mf parlai_internal/forward.ckpt \
	-bs 64 \
	-m transformer/generator \
	--ffn-size 512 \
	-esz 512 \
	-vtim 21600 \
	-sval True \
	-opt adam \
	-lr 0.1 \
	-df data/OpenSubtitles2018/opensubtitles.dict \
	--n-heads 4 \
	--dropout 0.1 \
	--attention-dropout 0.1 \
	-eps 10 \
	-ttim 86400
