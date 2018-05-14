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

    if len(sys.argv) != 2:
        print 'Invaild argments. Use: python create_model.py data_path'
        return
    path = sys.argv[1]
    model_path = os.path.join('model', path)

    # Prompt for confirm
    prompt_text = u'Create model will delete model that already exists (y/n)?:'
    if prompt(prompt_text).lower().strip() != 'y':
        print u'quit ...'
        return

    model_path_dot = '.'.join(model_path.split('/'))
    helper = importlib.import_module('%s.helper' % model_path_dot)
    config = importlib.import_module('%s.config' % model_path_dot)

    # Define the graph
    graph = tf.Graph()
    with graph.as_default():
        # Placenode for data feeding
        x = tf.placeholder(tf.int32, [None, config.max_length])
        y = tf.placeholder(tf.float32, [None, len(helper.classes)])
        keep_prob = tf.placeholder(tf.float32)

        # Embedding layer
        # embedding_dim = 64
        embedding = tf.get_variable('embedding', [len(helper.vocabulary) + 1, 64])
        embedding = tf.nn.embedding_lookup(embedding, x)

        # CNN layer
        conv = tf.layers.conv1d(embedding, filters=256, kernel_size=5)

        # Max pooling layer
        max_pooling = tf.reduce_max(conv, reduction_indices=[1])

        # Fully connected layer
        fully_connected = tf.layers.dense(max_pooling, units=128)
        fully_connected = tf.contrib.layers.dropout(fully_connected, keep_prob)
        fully_connected = tf.nn.relu(fully_connected)

        # Prediction function
        logits = tf.layers.dense(fully_connected, len(helper.classes))
        prediction = tf.argmax(tf.nn.softmax(logits), 1)

        # Loss function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(cross_entropy)

        # Optimizer function
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

        # Accuracy
        correct = tf.equal(tf.argmax(y, 1), prediction)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # Save model for using later
    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(session, os.path.join(model_path, 'model'))

    nodes = {
        'x': x.name,
        'y': y.name,
        'prediction': prediction.name,
        'keep_prob': keep_prob.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'accuracy': accuracy.name,
    }
    pickle.dump(nodes, open(os.path.join(model_path, 'nodes.pk'), 'wb'))


if __name__ == "__main__":
    main()
