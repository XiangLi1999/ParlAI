#!/bin/bash

python -m parlai.scripts.multiprocessing_train \
	-t dailydialog \
	-mf parlai_internal/forward_finetune.ckpt \
	-bs 16 \
	-m transformer/generator \
	-stim 21600 \
	-sval True \
	-opt sgd \
	-lr 0.1 \
	--embedding-type fasttext_cc \
	--beam-size 5 \
	-df data/OpenSubtitles2018/opensubtitles.dict \
	--dropout 0.1 \
	--attention-dropout 0.1 \
	-eps 10 \
	-ttim 43200 \
	-lfc True \
	--truncate 1024

