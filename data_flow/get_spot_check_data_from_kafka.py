#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2020/6/30 6:09 下午
@File    : get_spot_check_data_from_kafka.py
@Desc    : 从kafka队列获取谛听标注数据

"""

import kafka_handler
import time

# kafka.
kafka_user = "recommend_online"
kafka_password = "WGlQV0JCZUVXZg=="
kakfa_servers = "moli-kafka.prd.vivo.lan:9092"
kafka_topic = "feedscontent_push_tag_article"
kafka_group = "get_diting_tag"
kafka_result_consumer = kafka_handler.Consumer(kakfa_servers, kafka_user, kafka_password,
                                               kafka_topic, kafka_group)


def main():
    now_time = time.strftime("%Y-%m-%d", time.localtime())
    f = open("data_kafka_{}".format(now_time), "w+")
    for message in kafka_result_consumer.feed():
        try:
            f.write(message.decode("utf-8") + '\n')
        except Exception as e:
            print(e)
    f.close()


if __name__ == '__main__':
    main()

