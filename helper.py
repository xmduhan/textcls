#!/usr/bin/env python
# encoding: utf-8

import config
import pandas as pd
import tensorflow.contrib.keras as kr

classes = pd.read_csv('model/classes.csv', encoding='utf-8')
vocabulary = pd.read_csv('model/vocabulary.csv', encoding='utf-8')
charid = dict(vocabulary.set_index('char')['id'])
classid = dict(classes.set_index('class')['id'])


def content_to_vector(content):
    """ """
    alist = [[charid[c] for c in content if c in charid]]
    return kr.preprocessing.sequence.pad_sequences(alist, config.max_length)[0]


def class_to_vector(cls):
    """ """
    return kr.utils.to_categorical(classid[cls], num_classes=len(classes))[0]
