#!/usr/bin/env bash
# split by ','
source='dukemtmc,msmt17,cuhk03'
#source='dukemtmc'
#source='market1501'
target='market1501'
arch='resnet50_meta_v2'
epoch='90'
iter='200'
step_size='20'
batch_size='64'
ids='4'
dim='0'
mm='0.35'
bn='BN'
globall='1.0'
locall='1.0'
sscale='1'
python3 examples/meta_baseline_v2.py -ds ${source} \
-dt ${target} \
-a ${arch} \
--num-instances ${ids} \
--features ${dim} \
--mem-margin ${mm} \
--momentum '0.1' \
--bn-type ${bn} \
-cls \
--epochs ${epoch} -b ${batch_size} --iters ${iter} --step-size ${step_size} \
--bn-type ${bn} \
--global-lambda ${globall} \
--local-lambda ${locall} \
--logs-dir logs/meta_baseline/all-cls-l${locall}-g${globall}-ids${ids}-batch${batch_size} \