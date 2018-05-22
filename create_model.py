#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import pickle

import tensorflow as tf

from prompt_toolkit import prompt

import importlib

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def main():
    """ """

    if len(sys.argv) not in (2, 3):
        print 'Invaild argments. Use: python create_model.py data_path [graph_name]'
        return
    model_path = os.path.join('model', sys.argv[1])
    graph_name = sys.argv[2] if len(sys.argv) == 3 else 'cnn'

    # Prompt for confirm
    prompt_text = u'Create model will delete model that already exists (y/n)?:'
    if prompt(prompt_text).lower().strip() != 'y':
        print u'quit ...'
        return

    # Copy graph define to model path
    os.system('cp graph/%s.py %s/nn.py' % (graph_name, model_path))

    model_path_dot = '.'.join(model_path.split('/'))
    helper = importlib.import_module('%s.helper' % model_path_dot)
    config = importlib.import_module('%s.config' % model_path_dot)
    nn = importlib.import_module('%s.nn' % model_path_dot)
    graph, nodes = nn.create_graph(
        vocabulary=helper.vocabulary, classes=helper.classes, max_length=config.max_length)

    # Save model for using later
    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(session, os.path.join(model_path, 'model'))
    pickle.dump(nodes, open(os.path.join(model_path, 'nodes.pk'), 'wb'))


if __name__ == "__main__":
    main()
