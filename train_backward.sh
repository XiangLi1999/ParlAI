#!/bin/bash

python -m parlai.scripts.multiprocessing_train \
	-t opensubtitles \
	-mf parlai_internal/backward.ckpt \
	-bs 16 \
	-m transformer/generator \
	-stim 7200 \
	-sval True \
	-opt adam \
	-lr 1.0 \
	--embedding-type fasttext_cc \
	--beam-size 5 \
	--dropout 0.1 \
	--attention-dropout 0.1 \
	-eps 10 \
	-ttim 86400 \
	-lfc True \
	--truncate 1024 \
	-df data/OpenSubtitles2018/opensubtitles.dict \
