#!/usr/bin/env python
# encoding: utf-8

from __future__ import division


import pickle
from more_itertools import chunked
from collections import deque

import numpy as np
import pandas as pd

import tensorflow as tf

import importlib

from tqdm import tqdm

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
        average_loss = deque(maxlen=100)
        print_format = '[ %2d ]: loss: %.6f, train AC(%%): %.3f, test AC(%%): %.3f'
        for epoch in range(1, config.max_train_epoch):
            it = list(enumerate(chunked(range(train_length), 64), 1))
            pbar = tqdm(it, ncols=120, desc=print_format)
            for i, idxs in pbar:
                # Load sample data
                feed_data = train.loc[idxs]
                x_data = list(feed_data['content'].apply(helper.content_to_vector))
                y_data = list(feed_data['class'].apply(helper.class_to_vector))

                # Feed model
                eval_list = [optimizer, loss, accuracy]
                _, l, a = session.run(eval_list, feed_dict={x: x_data, y: y_data, keep_prob: config.keep_prob})
                average_accuracy.append(a)
                average_loss.append(l)

                # Print message
                pbar.set_description(print_format % (
                    epoch, np.mean(average_loss), np.mean(average_accuracy) * 100, np.nan))

                if i == len(it):
                    # Save model
                    saver.save(session, model_filename)

                    # Apply model to test data
                    cls = reload(importlib.import_module('%s.cls' % model_path_dot))
                    test = pd.read_csv(os.path.join(data_path, 'test.csv'), encoding='utf-8')
                    test = test.sample(1000)
                    test['prediction'] = test['content'].apply(cls.classify)
                    test['success'], test['count'] = (test['prediction'] == test['class']).astype(int), 1
                    test_accuracy = test['success'].sum() / test['count'].sum()
                    pbar.set_description(print_format % (
                        epoch, np.mean(average_loss), np.mean(average_accuracy) * 100, test_accuracy * 100))


if __name__ == "__main__":
    main()
