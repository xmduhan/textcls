#!/usr/bin/env python
# encoding: utf-8

import os
import pickle
import tensorflow as tf

from . import config
from . import helper


graph = tf.Graph()
tfcfg = tf.ConfigProto(intra_op_parallelism_threads=config.cpu_to_use)
session = tf.Session(graph=graph, config=tfcfg)
with session.graph.as_default():
    # Load model
    model_filename = os.path.join(helper.path, 'model')
    saver = tf.train.import_meta_graph(model_filename + '.meta')
    saver.restore(session, model_filename)

    # Load node information
    nodes = pickle.load(open(os.path.join(helper.path, 'nodes.pk'), "rU"))
    x = session.graph.get_tensor_by_name(nodes['x'])
    keep_prob = session.graph.get_tensor_by_name(nodes['keep_prob'])
    prediction = session.graph.get_tensor_by_name(nodes['prediction'])


def classify(content):
    """ """
    x_data = [helper.content_to_vector(content)]
    class_id = session.run(prediction, feed_dict={x: x_data, keep_prob: 1})[0]
    return helper.id_to_class[class_id]
