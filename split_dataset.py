#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    """ """
    if len(sys.argv) != 2:
        print 'Invaild argments. Use: python split_dataset.py data_path'
        return
    path = sys.argv[1]
    data_path = os.path.join('data', path)

    # Load train data
    filename = os.path.join(data_path, 'data.csv')
    print u'Reading data file: %s ....' % filename,
    df = pd.read_csv(filename, encoding='utf-8')
    df = df[['content', 'class']]
    print '(ok)'

    # Save splited dataset
    print 'Spliting ...',
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv(os.path.join(data_path, 'train.csv'), encoding='utf-8', index=False)
    test.to_csv(os.path.join(data_path, 'test.csv'), encoding='utf-8', index=False)
    print '(ok)'


if __name__ == "__main__":
    main()
