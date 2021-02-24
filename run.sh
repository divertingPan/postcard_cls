#!/usr/bin/env sh

python main.py \
--BATCHSIZE 256 \
--NETWORK resnet

python main.py \
--BATCHSIZE 200 \
--NETWORK senet

python main.py \
--BATCHSIZE 256 \
--NETWORK repvgg
