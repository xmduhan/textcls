#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import pandas as pd
from collections import Counter


def main():
    """ """
    if len(sys.argv) != 2:
        print 'Invaild argments. Use: python load_vocaublary.py data_path'
        return
    path = sys.argv[1]

    data_path = os.path.join('data', path)
    model_path = os.path.join('model', path)

    # Load train data
    train = pd.read_csv(os.path.join(data_path, 'train.csv'), encoding='utf-8')

    # mkdir
    if not os.path.exists(model_path):
        print 'Mkdir %s ... ' % model_path,
        os.makedirs(model_path)
        print '(ok)'

    # Generate vocabulary
    print u'Generating vocabulary ...',
    counter = Counter(''.join(list(train['content'])))
    vocabulary = pd.Series(counter).sort_values(ascending=False).reset_index()
    vocabulary.columns = ['char', 'count']
    vocabulary['id'] = range(1, len(vocabulary) + 1)
    vocabulary.to_csv(os.path.join(model_path, 'vocabulary.csv'), encoding='utf-8', index=False)
    print u'(ok)'

    # Generate classes
    print u'Generating classes ...',
    classes = train[['class']].drop_duplicates()
    classes['id'] = range(len(classes))
    classes.to_csv(os.path.join(model_path, 'classes.csv'), encoding='utf-8', index=False)
    print u'(ok)'


if __name__ == "__main__":
    main()
