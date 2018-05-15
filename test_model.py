#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import os
import sys
import pandas as pd
import importlib
from sklearn import metrics
from tabulate import tabulate

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def main():
    """ """
    if len(sys.argv) != 2:
        print 'Invaild argments. Use: python test_model.py data_path'
        return
    path = sys.argv[1]
    data_path = os.path.join('data', path)
    model_path = os.path.join('model', path)

    # Import config
    model_path_dot = '.'.join(model_path.split('/'))
    cls = importlib.import_module('%s.cls' % model_path_dot)

    # Load test data
    print u'Loading data ...',
    df = pd.read_csv(os.path.join(data_path, 'test.csv'), encoding='utf-8')
    print u'(ok)'

    # Apply model to predict
    print 'Applying model ...',
    df['prediction'] = df['content'].apply(cls.classify)
    print u'(ok)'

    # Print predicting result
    print u'Predicting result:'
    df['success'], df['count'] = (df['prediction'] == df['class']).astype(int), 1
    result = df.groupby('prediction')[['count', 'success']].sum()
    result['accuracy'] = result['success'] / result['count']
    print tabulate(result, headers='keys', tablefmt='psql')
    print 'Total Accuracy:', result['success'].sum() / result['count'].sum()

    # Print confusion matrix
    print u'Confusion Matrix:'
    print tabulate(metrics.confusion_matrix(df['class'], df['prediction']))


if __name__ == "__main__":
    main()
