#!/usr/bin/env python
# encoding: utf-8

import config
import pickle
import tensorflow as tf

from helper import content_to_vector, id_to_class


graph = tf.Graph()
tfcfg = tf.ConfigProto(intra_op_parallelism_threads=config.cpu_to_use)
session = tf.Session(graph=graph, config=tfcfg)
with session.graph.as_default():
    # Load model
    saver = tf.train.import_meta_graph(config.model_filename + '.meta')
    saver.restore(session, config.model_filename)

    # Load node information
    nodes = pickle.load(open(config.nodes_filename, "rU"))
    x = session.graph.get_tensor_by_name(nodes['x'])
    keep_prob = session.graph.get_tensor_by_name(nodes['keep_prob'])
    prediction = session.graph.get_tensor_by_name(nodes['prediction'])


def classify(content):
    """ """
    x_data = [content_to_vector(content)]
    class_id = session.run(prediction, feed_dict={x: x_data, keep_prob: .7})[0]
    return id_to_class[class_id]
