#!/usr/bin/env bash
# split by ','
source='dukemtmc'
#source='dukemtmc'
#source='market1501'
target='market1501'
arch='resnet50_plus'
python3 examples/uda_baseline.py -ds ${source} -dt ${target} -a ${arch} \
--logs-dir logs/UDA_baseline/base
