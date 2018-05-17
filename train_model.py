#!/usr/bin/env python
# encoding: utf-8

from __future__ import division


import pickle
from datetime import datetime
from more_itertools import chunked
from collections import deque

import numpy as np
import pandas as pd

import tensorflow as tf

import importlib

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def main():
    """ """
    if len(sys.argv) != 2:
        print 'Invaild argments. Use: python train_model.py data_path'
        return
    path = sys.argv[1]

    data_path = os.path.join('data', path)
    model_path = os.path.join('model', path)

    # Import config
    model_path_dot = '.'.join(model_path.split('/'))
    config = importlib.import_module('%s.config' % model_path_dot)
    helper = importlib.import_module('%s.helper' % model_path_dot)

    # Load Training data
    train = pd.read_csv(os.path.join(data_path, 'train.csv'), encoding='utf-8')
    train = train.sample(len(train)).reset_index(drop=True)
    train_length = len(train)

    # Initial Tensorflow
    graph = tf.Graph()
    tfcfg = tf.ConfigProto(intra_op_parallelism_threads=config.cpu_to_use)
    session = tf.Session(graph=graph, config=tfcfg)
    with session.graph.as_default():

        # Load model
        model_filename = os.path.join(model_path, 'model')
        saver = tf.train.import_meta_graph(model_filename + '.meta')
        saver.restore(session, model_filename)

        # Load model nodes information
        nodes = pickle.load(open(os.path.join(model_path, 'nodes.pk'), "rU"))
        x = session.graph.get_tensor_by_name(nodes['x'])
        y = session.graph.get_tensor_by_name(nodes['y'])
        keep_prob = session.graph.get_tensor_by_name(nodes['keep_prob'])
        loss = session.graph.get_tensor_by_name(nodes['loss'])
        accuracy = session.graph.get_tensor_by_name(nodes['accuracy'])
        optimizer = session.graph.get_operation_by_name(nodes['optimizer'])

        # Training ...
        average_accuracy = deque(maxlen=100)
        for epoch in range(1, config.max_train_epoch):
            for i, idxs in enumerate(chunked(range(train_length), 64), 1):

                # Load sample data
                feed_data = train.loc[idxs]
                x_data = list(feed_data['content'].apply(helper.content_to_vector))
                y_data = list(feed_data['class'].apply(helper.class_to_vector))

                # Feed model
                begin_time = datetime.now()
                eval_list = [optimizer, loss, accuracy]
                _, l, a = session.run(eval_list, feed_dict={x: x_data, y: y_data, keep_prob: config.keep_prob})
                average_accuracy.append(a)

                print '%d - %5d: loss: %.6f, accuracy: %.3f, average accuracy: %.3f, time: %.2fs' % (
                    epoch, i, l, a, np.mean(average_accuracy), (datetime.now() - begin_time).total_seconds())

                # Save model
                saver.save(session, model_filename)


if __name__ == "__main__":
    main()
