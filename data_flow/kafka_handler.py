#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2020/7/7 1:45 下午
@File    : kafka_handler.py
@Desc    : 

"""

import sys
import time
import json
from confluent_kafka import Producer as ConfluentKafkaProducer
from confluent_kafka import Consumer as ConfluentKafkaConsumer
# from kafka import KafkaConsumer
# from kafka.errors import KafkaError
import time


class Producer(object):
    """
    生产模块
    """

    def __init__(self, bootstrap_servers, user, password, kafka_topic):  # 传参.

        self.bootstrap_servers = bootstrap_servers
        self.user = user
        self.password = password
        self.kafka_topic = kafka_topic

        self.producer = ConfluentKafkaProducer({
            'bootstrap.servers': self.bootstrap_servers,
            'security.protocol': "SASL_PLAINTEXT",
            'sasl.mechanism': 'SCRAM-SHA-256',
            'sasl.username': self.user,
            'sasl.password': self.password})

    def send_bulk(self, messages):
        for message in messages:
            self.send(message)
        self.flush()

    def send(self, message):  # 函数名用动词.
        self.producer.produce(self.kafka_topic, value=message)

    def flush(self):
        self.producer.flush()


class Consumer(object):
    """
    消费模型
    """

    def __init__(self, bootstrap_servers, user, password, kafka_topic, group_id):
        self.bootstrap_servers = bootstrap_servers
        self.user = user
        self.password = password
        self.group_id = group_id
        self.kafka_topic = kafka_topic
        self.consumer = ConfluentKafkaConsumer({
            'bootstrap.servers': self.bootstrap_servers,
            'security.protocol': "SASL_PLAINTEXT",
            'sasl.mechanism': 'SCRAM-SHA-256',
            'sasl.username': self.user,
            'sasl.password': self.password,
            'session.timeout.ms': '15000',
            'heartbeat.interval.ms': '5000',
            'max.poll.interval.ms': '60000',
            'group.id': self.group_id,
            'client.id': self.user})
        self.consumer.subscribe([kafka_topic])

    def feed(self):
        while 1:
            msg = self.consumer.poll(timeout=3000)
            if msg is None or msg.value() is None:
                time.sleep(0.01)
                continue
            if msg.error():
                continue
            yield msg.value()


# def main():
#     consumer = get_consumer_task_kafka()
#     print('consumer:', consumer)
#     while True:
#         msg = consumer.poll(timeout=3000)
#         if msg is None or msg.value() is None:
#             time.sleep(0.01)
#             continue
#         if msg.error():
#             logger.warning("msg.error: {}".format(msg.error()))
#             continue
#         logger.info("msg.value: {}".format(msg.value()))
#         print(msg.value())
#         continue
#
#
# if __name__ == '__main__':
#     main()
