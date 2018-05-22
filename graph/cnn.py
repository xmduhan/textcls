#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def create_graph(vocabulary, classes, max_length):
    """
    vocabulary  vocabulary of text
    classess  classes to take
    max_length  max of length text
    """
    # Define the graph
    graph = tf.Graph()
    with graph.as_default():
        # Placenode for data feeding
        x = tf.placeholder(tf.int32, [None, max_length])
        y = tf.placeholder(tf.float32, [None, len(classes)])
        keep_prob = tf.placeholder(tf.float32)

        # Embedding layer
        # embedding_dim = 64
        embedding = tf.get_variable('embedding', [len(vocabulary) + 1, 64])
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
        logits = tf.layers.dense(fully_connected, len(classes))
        prediction = tf.argmax(tf.nn.softmax(logits), 1)

        # Loss function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(cross_entropy)

        # Optimizer function
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

        # Accuracy
        correct = tf.equal(tf.argmax(y, 1), prediction)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    nodes = {
        'x': x.name,
        'y': y.name,
        'prediction': prediction.name,
        'keep_prob': keep_prob.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'accuracy': accuracy.name,
    }

    return graph, nodes
