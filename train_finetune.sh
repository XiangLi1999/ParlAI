#!/bin/bash

python -u examples/train_model.py \
	-t dailydialog \
	-mf parlai_internal/forward_finetune.ckpt \
	-bs 16 \
	-m transformer/generator \
	--ffn-size 256 \
	-esz 256 \
	-stim 7200 \
	-sval True \
	-opt adam \
	-lr 1 \
	-df data/OpenSubtitles2018/opensubtitles.dict \
	--n-heads 2 \
	--dropout 0.1 \
	--attention-dropout 0.1 \
	-eps 10 \
	-ttim 43200 \
	-lfc True \
	--truncate 1024

