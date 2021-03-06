#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2020/6/22 5:58 下午
@File    : run_train.py
@Desc    : 训练入口

"""

from config import Config
from train import Train
import os
import shutil
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = Config(config_file=sys.argv[1])

batch_size_list = [64]
# 文本token长度在1000以下的比例: 0.9696374110471027
# 文本char长度在2000以下的比例: 0.9327188523664295
val_len = [1000]

learning_rate_list = [0.005, 0.0001]
drop_list = [0.8]
for batch_size in batch_size_list:
    for token_len in val_len:
        for learning_rate in learning_rate_list:
            for drop in drop_list:
                # if os.path.exists("model_save/" + "batch_size_" + str(config.train.batch_size)
                #    + 'token_len' + str(token_len)
                #    +'_lr_'+str(learning_rate)+'drop_'+str(drop)):
                config.var_len_feature.max_var_token_length = token_len
                config.var_len_feature.max_var_char_length = token_len * 2
                config.train.hidden_layer_dropout_keep_prob = drop
                config.train.batch_size = batch_size
                config.train.learning_rate = learning_rate

                model_path = "batch_size_{}lr_{}drop_{}_fasttext_v1".format(config.train.batch_size,
                                                                            config.train.learning_rate, drop)

                if os.path.exists("/data222/11121175/classifier_result/model_save_best/{}".format(model_path)):
                    continue

                Train(config)
                f = open('Best.txt')
                lines = f.readlines()
                f.close()
                idx_model = int(lines[-1].split('\t')[1])

                files = [str(file_name) for file_name in sorted([int(file) for file in os.listdir('export_model/')])]

                path = files[idx_model - 1]

                shutil.copytree("export_model/", "/data222/11121175/classifier_result/model_all/{}".format(model_path))
                shutil.copytree("export_model/{}".format(path), "/data222/11121175/classifier_result/model_save_best/{}".format(model_path))
                shutil.copytree("eval_dir/", "/data222/11121175/classifier_result/result_save/{}".format(model_path))
                os.remove('Best.txt')
                # clean data
                shutil.rmtree('export_model', ignore_errors=True)
                # shutil.rmtree('tfrecord', ignore_errors=True)
                shutil.rmtree('eval_dir', ignore_errors=True)
                shutil.rmtree('checkpoint', ignore_errors=True)
                shutil.rmtree('__pycache__', ignore_errors=True)
