#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2020/6/16 5:15 下午
@File    : run_get_vec.py
@Desc    : 

"""

from .test_we import redis_conn, wrap_key
import json
import sys
import pandas as pd
import numpy as np
import multiprocessing


def read_redis(res):
    res = _redis.get(wrap_key(res))
    if res == None:
        return None
    res = json.loads(res)
    res = ' '.join([str(item) for item in res])
    return res


def get_vec(data_part, i, dict_txt):
    data_part[1] = data_part.apply(lambda x: read_redis(x[0]), axis=1)
    data_part = data_part.dropna(how='any', axis=0)
    data_part.to_csv(dict_txt + '_' + str(i), sep='\t', index=None, header=None, mode='a')


if __name__ == "__main__":
    _redis = redis_conn()
    # file=open('token_all.dict')
    # file2=open('token_vec_all.txt','w+')
    dict_now = sys.argv[1]
    dict_txt = sys.argv[2]
    data = pd.read_csv(dict_now, sep='\t', quoting=3, header=None)
    len_data = len(data)

    pros = []
    for i in range(30):
        if i < 29:
            data_temp = data.iloc[i * len(data) // 30:(i + 1) * len(data) // 30].copy()
        else:
            data_temp = data.iloc[i * len(data) // 30:len(data)].copy()
        p = multiprocessing.Process(target=get_vec, args=(data_temp, i, dict_txt))
        pros.append(p)

    for item in pros:
        item.start()
    for item in pros:
        item.join()
