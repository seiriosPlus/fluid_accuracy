#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export CPU_NUM=4
export BASE=/accuracy/

cd /accuracy/text_classification/

if [ "$1" = "local" ]
then
    GLOG_v=0 GLOG_logtostderr=1 stdbuf -oL python train_high_api.py  --dict_path /root/.cache/paddle/dataset/imdb/imdb.vocab  --local 1 &> $BASE/trainerstand.log &
    exit 0
fi

export PADDLE_PSERVER_PORT=36001
export PADDLE_PSERVER_IPS=127.0.0.1
export PADDLE_TRAINERS=1
export PADDLE_CURRENT_IP=127.0.0.1
export PADDLE_TRAINER_ID=0

if [ "$1" = "ps" ]
then
    export PADDLE_TRAINING_ROLE=PSERVER
     
    export GLOG_vi=0
    export GLOG_logtostderr=1

    echo "PADDLE WILL START PSERVER ..."
    stdbuf -oL python train_high_api.py  --dict_path /root/.cache/paddle/dataset/imdb/imdb.vocab --local 0 &> $BASE/pserver_1x1.log &
fi

if [ "$1" = "tr" ]
then
    export PADDLE_TRAINING_ROLE=TRAINER

    export GLOG_v=0
    export GLOG_logtostderr=1

    echo "PADDLE WILL START TRAINER ..."
    stdbuf -oL python train_high_api.py  --dict_path /root/.cache/paddle/dataset/imdb/imdb.vocab --local 0 &> $BASE/trainer_1x1.log &
fi
