#!/usr/bin/env bash
# split by ','
source='market1501'
target='market1501'
arch='resnet50_plus'
epoch='90'
batch_size='256'
iter='100'
step_size='30'
python3 examples/reid_baseline.py -ds ${source} \
-dt ${target} \
-a ${arch} \
--epochs ${epoch} -b ${batch_size} --iters ${iter} --step-size ${step_size} \
--logs-dir logs/ReID_baseline/step_epo${epoch}-step${step_size}-iter${iter}-batch${batch_size} \
