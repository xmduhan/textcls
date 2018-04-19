#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import config

import pickle
from datetime import datetime
from more_itertools import chunked
from collections import deque

import numpy as np
import pandas as pd

import tensorflow as tf

from helper import content_to_vector, class_to_vector

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def main():
    """ """
    # Load Training data
    train = pd.read_csv('data/train.csv', encoding='utf-8')
    train = train.sample(len(train)).reset_index(drop=True)
    train_length = len(train)

    # Initial Tensorflow
    graph = tf.Graph()
    tfcfg = tf.ConfigProto(intra_op_parallelism_threads=config.cpu_to_use)
    session = tf.Session(graph=graph, config=tfcfg)
    with session.graph.as_default():

        # Load model
        saver = tf.train.import_meta_graph(config.model_filename + '.meta')
        saver.restore(session, config.model_filename)

        # Load model nodes information
        nodes = pickle.load(open(config.nodes_filename, "rU"))
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
                x_data = list(feed_data['content'].apply(content_to_vector))
                y_data = list(feed_data['class'].apply(class_to_vector))

                # Feed model
                begin_time = datetime.now()
                eval_list = [optimizer, loss, accuracy]
                _, l, a = session.run(eval_list, feed_dict={x: x_data, y: y_data, keep_prob: .75})
                average_accuracy.append(a)

                print '%d - %5d: loss: %.6f, accuracy: %.3f, average accuracy: %.3f, time: %.2fs' % (
                    epoch, i, l, a, np.mean(average_accuracy), (datetime.now() - begin_time).total_seconds())

                # Save model
                saver.save(session, config.model_filename)


if __name__ == "__main__":
    main()
