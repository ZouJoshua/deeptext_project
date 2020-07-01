#!/usr/bin/env python
# coding:utf8

# Copyright (c) 2018, Tencent. All rights reserved

# Predictor that load one or multi model that save by Estimators
#   and predict the given text.
# Usage: python predict.py config.json
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
import json
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import reader
# from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_def_utils


import util
from config import Config
from data_processor import DataProcessor


class Predictor(object):
    def __init__(self, config):
        self.config = config
        self.sess_model_list = []
        self.graph_list = []
        self.signature_def_list = []
        self._read_sessions(self.config.predict.model_dirs,
                            self.config.predict.model_tag)
        self.model_weights = []
        for model_weight in self.config.predict.model_weights:
            self.model_weights.append(float(model_weight))
        assert len(self.model_weights) == len(self.sess_model_list)

        self.data_processor = DataProcessor(config)
        self.data_processor.load_all_dict()

        self.feature_debug_file = codecs.open("feature_debug.txt", "w",
                                              encoding=util.CHARSET)

    def _read_sessions(self, model_dirs, tag):
        """Read graph and parameters.
        Args:
            model_dirs: Model dirs saved by Estimator
            tag: Serving tag e.g. serve
        """
        for model_dir in model_dirs:
            saved_model = reader.read_saved_model(model_dir)
            meta_graph = None
            for meta_graph_def in saved_model.meta_graphs:
                if tag in meta_graph_def.meta_info_def.tags:
                    meta_graph = meta_graph_def
                    break
            if meta_graph is None:
                raise ValueError("Cannot find saved_model with tag: " + tag)
            self.signature_def_list.append(
                get_signature_def_by_key(meta_graph, "probs"))

            gpu_option = tf.GPUOptions(
                allow_growth=True,
                visible_device_list=self.config.train.visible_device_list)
            session_config = tf.ConfigProto(
                gpu_options=gpu_option)
            graph = tf.Graph()
            self.graph_list.append(graph)
            session = tf.Session(graph=graph, config=session_config)
            self.sess_model_list.append(session)
            with session.as_default():
                with graph.as_default():
                    loader.load(session, [tag], model_dir)
                    graph.finalize()

    @staticmethod
    def _single_model_predict(signature_def, sess_model, examples):
        """Predict example
        Args:
            signature_def: Signature_def of the model.
            sess_model: Session to predict with.
            examples: Tf.example to predict.
        returns:
            probs
        """
        with sess_model.as_default():
            with sess_model.graph.as_default():
                inputs_feed_dict = {
                    signature_def.inputs["inputs"].name: examples,
                }
                output_tensor = signature_def.outputs["scores"].name
        print('here')
        return sess_model.run(output_tensor, feed_dict=inputs_feed_dict)

    def predict(self, texts):
        """Predict example. If one model, output probs.
        If multi model, output the model ensemble result.
        Args:
            text: Text to predict.
        returns:
             If one model, output probs.
             If multi model, output the model ensemble result.
        """
        examples = []
        for text in texts:
            example, feature_sample = \
                    self.data_processor.get_tfexample_from_text(text, False)

            if example is None:
                return None
            feature_str = json.dumps(feature_sample, ensure_ascii=False)
            self.feature_debug_file.write(feature_str + "\n")
            example = example.SerializeToString()
            examples.append(example)

        if len(self.sess_model_list) == 1:
            # probs
            outputs = self._single_model_predict(
                self.signature_def_list[0], self.sess_model_list[0], examples)
        else:
            # vote counts
            outputs = np.zeros(
                    [len(texts), 1, len(self.data_processor.label_map)])
            for i in range(len(self.sess_model_list)):
                scores = self._single_model_predict(
                    self.signature_def_list[i], self.sess_model_list[i],
                    examples)
                print('here_over')
                for k in range(len(scores)):
                    outputs[k] += scores[k] * self.model_weights[i]

        return outputs

def get_signature_def_by_key(meta_graph_def, signature_def_key):
    """Utility function to get a SignatureDef protocol buffer by its key.
    Args:
      meta_graph_def: MetaGraphDef protocol buffer with the SignatureDefMap to
        look up.
      signature_def_key: Key of the SignatureDef protocol buffer to find in the
        SignatureDefMap.
    Returns:
      A SignatureDef protocol buffer corresponding to the supplied key, if it
      exists.
    Raises:
      ValueError: If no entry corresponding to the supplied key is found in the
      SignatureDefMap of the MetaGraphDef.
    """
    if signature_def_key not in meta_graph_def.signature_def:
        raise ValueError("No SignatureDef with key '%s' found in MetaGraphDef." %
                         signature_def_key)
    return meta_graph_def.signature_def[signature_def_key]


def main(_):

    config = Config(config_file=sys.argv[1])
    outputs = Predictor(config)


if __name__ == '__main__':
    tf.app.run()
