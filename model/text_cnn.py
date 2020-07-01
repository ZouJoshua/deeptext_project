#!/usr/bin/env python
# coding:utf8

# Copyright (c) 2018, Tencent. All rights reserved


# References "Convolutional Neural Networks for Sentence Classification"
#            "A Sensitivity Analysis of (and Practitioners’ Guide to)
#             Convolutional Neural Networks for Sentence Classification"

import tensorflow as tf

from model.embedding_layer import EmbeddingLayer
from model.model_helper import ModelHelper


class TextCNNEstimator(tf.estimator.Estimator):
    def __init__(self, data_processor, model_params):
        config = data_processor.config
        embedding_layer = EmbeddingLayer(config)
        model_helper = ModelHelper(config)

        def _model_fn(features, labels, mode, params):
            self._check(params["feature_names"], data_processor)
            feature_name = params["feature_names"][0]
            index = data_processor.dict_names.index(feature_name)
            sequence_length = data_processor.max_sequence_length[index] + \
                              config.fixed_len_feature.token_padding_begin + \
                              config.fixed_len_feature.token_padding_end
            padding_value = \
                data_processor.token_map[data_processor.VOCAB_PADDING]
            embedding = embedding_layer.get_vocab_embedding(
                feature_name, features["fixed_len_" + feature_name],
                len(data_processor.dict_list[index]), params["epoch"],
                pretrained_embedding_file=
                data_processor.pretrained_embedding_files[index],
                dict_map=data_processor.dict_list[index],
                mode=mode,
                begin_padding_size=config.fixed_len_feature.token_padding_begin,
                end_padding_size=config.fixed_len_feature.token_padding_end,
                padding_id=padding_value)
            embedding = tf.expand_dims(embedding, -1)

            if mode == tf.estimator.ModeKeys.TRAIN:
                embedding = model_helper.dropout(
                    embedding,
                    config.embedding_layer.embedding_dropout_keep_prob)

            filter_sizes = config.TextCNN.filter_sizes
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("convolution-max_pooling-%d" % filter_size):
                    filter_shape = \
                        [filter_size,
                         config.embedding_layer.embedding_dimension,
                         1, config.TextCNN.num_filters]
                    W = tf.Variable(
                        tf.random_uniform(filter_shape, minval=-0.01,
                                          maxval=0.01),
                        name="W-%d" % filter_size)
                    b = tf.get_variable("b-%d" % filter_size,
                                        [config.TextCNN.num_filters])
                    # Strides is set to [1, 1, 1, 1].
                    # Convolution will slide 1 vocab at one time
                    convolution = tf.nn.conv2d(
                        embedding, W, strides=[1, 1, 1, 1],
                        padding="VALID", name="convolution")
                    h = tf.nn.relu(tf.nn.bias_add(convolution, b), name="relu")
                    pooled = tf.nn.max_pool(
                        h, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1], padding='VALID',
                        name="max_pooling")
                    pooled_outputs.append(pooled)

            num_filters_total = config.TextCNN.num_filters * len(filter_sizes)
            # pooled_outputs contains
            # tensor with shape [batch_size, 1, 1, num_filters]
            h_pool = tf.concat(pooled_outputs, 3)
            hidden_layer = tf.reshape(h_pool, [-1, num_filters_total])

            if mode == tf.estimator.ModeKeys.TRAIN:
                hidden_layer = model_helper.dropout(
                    hidden_layer, config.train.hidden_layer_dropout_keep_prob)
                # when repeating the result in the paper, the following code
                # should be added.
                # hidden_layer *= FLAGS.hidden_layer_dropout_keep_prob * (
                #         1 - FLAGS.hidden_layer_dropout_keep_prob)

            return model_helper.get_softmax_estimator_spec(
                hidden_layer, mode, labels, params["label_size"],
                params["static_embedding"], data_processor.label_dict_file)

        super(TextCNNEstimator, self).__init__(
            model_fn=_model_fn, model_dir=config.model_common.checkpoint_dir,
            config=model_helper.get_run_config(), params=model_params)

    @staticmethod
    def _check(feature_names, data_processor):
        # TextCNN only support one feature and should be char or token.
        assert len(feature_names) == 1
        assert feature_names[0] in data_processor.dict_names
        index = data_processor.dict_names.index(feature_names[0])
        assert len(data_processor.dict_list[index]) > 0
        assert data_processor.max_sequence_length[index] > 0
