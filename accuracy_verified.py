# -*- coding: UTF-8 -*-

#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Accuracy verified in Fluid"""

import sys
import os

import re

import numpy as np
np.set_printoptions(precision=8, suppress=True, formatter={'float_kind':'{:0.8f}'.format})

INDEX_MAP = {1:'Loss', 2:'Accuracy', 3:'Test Loss', 4:'Test Accuracy'}

STANDALONG_TRAIN_LOG = None 
CLUSTER_TRAIN_LOGS = None


def get_absolute_path(log_regex_name, dirname):
    assert os.path.isdir(dirname)

    paths = []

    files = os.listdir(dirname)

    for f in files:
        matchs = re.match(log_regex_name, f)

        if matchs:
            paths.append(os.path.join(dirname, f))

    return paths
        

def get_standalone_log(log_regex_name="trainerstand.log", dirname=None):
    dirname = dirname if dirname else os.getcwd()
    paths = get_absolute_path(log_regex_name, dirname)
    
    assert paths is not None and len(paths) == 1
    return paths[0]


def get_cluster_logs(log_regex_name="trainer\.\d\.log", dirname=None):
    dirname = dirname if dirname else os.getcwd()
    paths = get_absolute_path(log_regex_name, dirname)
    assert len(paths) >= 1
    return paths


def read_log(log_file_name):
    train_datas = {} 
    with open(log_file_name, 'r') as logs:
        for log in logs.readlines():

            if "Accuracy" and "Step" not in log:
                continue

            train_data = []
            log = log.strip()

            groups = log.split(",")
            assert len(groups) == 7

            iter=[]
            for group in groups[0:2]:
                group = group.strip()
                val = group.split(":")[1].strip()
                iter.append(val)
            train_data.append("_".join(iter))

            for group in groups[2:]:
                group = group.strip()
                val = group.split(":")[1]
                train_data.append(float(val))

            assert len(train_data) == 6 
            train_datas[train_data[0]] = train_data[1:]

    return train_datas


def stat_train_logs():
    standalone_log = read_log(STANDALONG_TRAIN_LOG)
    cluster_logs = []
    for log in CLUSTER_TRAIN_LOGS:
        cluster_log = read_log(log)
        cluster_logs.append(cluster_log)

    return (standalone_log, cluster_logs) 


def index_calculation(standalone_log, cluster_logs, index, name):
    standalone_log = np.array(standalone_log)
    cluster_logs = np.array(cluster_logs)

    standalone_val = (standalone_log[..., index]).mean()
    cluster_val = (cluster_logs[..., index]).mean()

    return standalone_val, cluster_val, abs(standalone_val - cluster_val)


def global_calc(standalone_log, cluster_logs, epoch, step):
    itea = "{}_{}".format(epoch, step)

    standalone_itea = standalone_log.get(itea)
    assert standalone_itea is not None

    cluster_iteas = []
    for cluster_log in cluster_logs:
        cluster_itea = cluster_log.get(itea)
        assert cluster_itea is not None
        cluster_iteas.append(cluster_itea)

    print("")
    print("FOR EPOCH: {} AND STEP: {}".format(epoch, step))
    for index,name in INDEX_MAP.items():
        standalone_val, cluster_val, error = index_calculation(standalone_itea, cluster_iteas, index, name)
        print("NAME: {}, STANDALONE: {}, CLUSER MEAN: {}, ERROR: {}".format(name, standalone_val, cluster_val, error))
    print("")


def specific_calc(standalone_log, cluster_logs, index):

    def numeric_compare(x, y):
        x1,x2 = x.split("_")
        y1,y2 = y.split("_")

        return int(x2)-int(y2) if int(x1) == int(y1) else int(x1)-int(y1)

    iteas = standalone_log.keys()
    iteas.sort(cmp=numeric_compare, reverse=False)

    for itea in iteas:
        standalone_itea = standalone_log.get(itea)
        cluster_iteas = []
        for cluster_log in cluster_logs:
            cluster_itea = cluster_log.get(itea)
            if cluster_itea is None:
                break
            cluster_iteas.append(cluster_itea)

        if standalone_itea is None or len(cluster_iteas) != len(CLUSTER_TRAIN_LOGS):
            break
        
        name = INDEX_MAP.get(index)
        standalone_val, cluster_val, error = index_calculation(standalone_itea, cluster_iteas, index, name)
        print("ITEA: {}, STANDALONE: {}, CLUSER MEAN: {}, ERROR: {}".format(itea, standalone_val, cluster_val, error))

 
if __name__=="__main__":

    basedir = sys.argv[1]
    epoch = int(sys.argv[2])
    step = int(sys.argv[3])

    STANDALONG_TRAIN_LOG = get_standalone_log(dirname=basedir) 
    CLUSTER_TRAIN_LOGS = get_cluster_logs(dirname=basedir)

    print("STANDALONG_TRAIN_LOG: {}".format(STANDALONG_TRAIN_LOG))
    print("CLUSTER_TRAIN_LOGS  : {}".format(CLUSTER_TRAIN_LOGS))


    index = None

    if len(sys.argv) > 4:
        index = int(sys.argv[4])

    standalone_log, cluster_logs = stat_train_logs()

    if index:
        specific_calc(standalone_log, cluster_logs, index)
    else:
        global_calc(standalone_log, cluster_logs, epoch, step)
    




