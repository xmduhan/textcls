#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
from collections import Counter


def main():
    """ """

    # Load train data
    train = pd.read_csv('data/train.csv', encoding='utf-8')

    # Generate vocabulary
    counter = Counter(''.join(list(train['content'])))
    vocabulary = pd.Series(counter).sort_values(ascending=False).reset_index()
    vocabulary.columns = ['char', 'count']
    vocabulary['id'] = range(1, len(vocabulary) + 1)
    vocabulary.to_csv('model/vocabulary.csv', encoding='utf-8', index=False)

    # Generate classes
    classes = train[['class']].drop_duplicates()
    classes['id'] = range(len(classes))
    classes.to_csv('model/classes.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    main()
