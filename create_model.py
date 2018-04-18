#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import os
import pickle

import pandas as pd
import tensorflow as tf

from collections import Counter

from prompt_toolkit import prompt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    """ """
    # Prompt for confirm
    prompt_text = u'Create model will delete model that already exists (y/n)?'
    if prompt(prompt_text).lower().strip() != 'y':
        print u'quit ...'
        return

    # Load train data
    train = pd.read_csv('data/train.csv', encoding='utf-8')

    # Generate vocabulary
    counter = Counter(''.join(list(train['content'])))
    vocabulary = pd.Series(counter).sort_values(ascending=False).reset_index()
    vocabulary.columns = ['char', 'count']
    vocabulary.to_csv('model/vocabulary.csv', encoding='utf-8')

    # Generate classes
    classes = train['class'].drop_duplicates()
    classes = classes.reset_index(drop=True)
    classes.columns = ['class']
    classes.to_csv('model/classes.csv', encoding='utf-8')

    # Define the graph
    graph = tf.Graph()
    with graph.as_default():
        # Placenode for data feeding
        # seq_length = 600
        x = tf.placeholder(tf.int32, [None, 600])
        y = tf.placeholder(tf.float32, [None, len(classes)])
        keep_prob = tf.placeholder(tf.float32)

        # Embedding layer
        # embedding_dim = 64
        embedding = tf.get_variable('embedding', [len(vocabulary), 64])
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
        correct = tf.equal(tf.argmax(x, 1), prediction)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        model_path = 'model'
        model_filename = 'model/model'

    # Save model for using later
    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(session, model_filename)

    nodes = {
        'x': x.name,
        'y': y.name,
        'prediction': prediction.name,
        'keep_prob': keep_prob.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'accuracy': accuracy.name,
    }
    nodes_file_name = os.path.join(model_path, 'nodes.pk')
    pickle.dump(nodes, open(nodes_file_name, 'wb'))


if __name__ == "__main__":
    main()
