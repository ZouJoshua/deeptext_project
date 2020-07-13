#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2020/6/16 5:21 下午
@File    : test_we.py
@Desc    : 

"""

from rediscluster import StrictRedisCluster
import sys
import json
import numpy as np


def redis_conn():
    redis_nodes = [{'host': 'feedsad-user-online-behavior4.redis.dba.vivo.lan.', 'port': 11104},
                   {'host': 'feedsad-user-online-behavior3.redis.dba.vivo.lan.', 'port': 13400},
                   {'host': 'feedsad-user-online-behavior2.redis.dba.vivo.lan.', 'port': 11103},
                   {'host': 'feedsad-user-online-behavior1.redis.dba.vivo.lan.', 'port': 13399},
                   {'host': 'feedsad-user-online-behavior0.redis.dba.vivo.lan.', 'port': 11101}]
    try:
        redisconn = StrictRedisCluster(startup_nodes=redis_nodes)
    except Exception as e:
        print('Connect Error, %s' % (e,))
        sys.exit()
    return redisconn


def wrap_key(key):
    return 'we:v0:{%s}' % (key,)


def cosine(x1, x2):
    v1, v2 = np.array(x1), np.array(x2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * (np.linalg.norm(v2)))


if __name__ == '__main__':

    args = sys.argv[1:]
    if not args:
        print('No words.')
        exit()

    _redis = redis_conn()
    if len(args) == 1:
        word = args[0]
        res = _redis.get(wrap_key(word))
        if not res:
            print('can not find word: %s' % (word,))
        print(json.loads(res))
    elif len(args) == 2:
        w1, w2 = args[0], args[1]
        x1, x2 = json.loads(_redis.get(wrap_key(w1))), json.loads(_redis.get(wrap_key(w2)))
        cos_value = cosine(x1, x2)
        print(cos_value)
    else:
        print('More than 2 words.')
