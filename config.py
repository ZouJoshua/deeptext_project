#!/usr/bin/env python
# coding:utf8

# Copyright (c) 2018, Tencent. All rights reserved

# Load config
# Use json config to support server load different models,
# e.g. models trained from different train data.

import json


class Config(object):
    """Config load from json file
    """

    def __init__(self, config=None, config_file=None):
        if config_file:
            with open(config_file, 'r') as fin:
                config = json.load(fin)

        if config:
            self._update(config)

    def add(self, key, value):
        """add
        Args:
                key(type):
                value(type):
        Returns:
                type:
        """
        self.__dict__[key] = value

    def _update(self, config):
        if not isinstance(config, dict):
            return

        for key in config:
            if isinstance(config[key], dict):
                config[key] = Config(config[key])

            if isinstance(config[key], list):
                config[key] = [Config(x) if isinstance(x, dict) else x for x in
                               config[key]]

        self.__dict__.update(config)
