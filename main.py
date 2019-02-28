# -*- coding: utf-8 -*-

# TODO:
## Active learning algorithms:
# MS (margin sampling, o most uncertain)
# MCLU (multi-class label uncertainty)
# SSC (significance space construction)
# nEQB (normalized entropy query bagging)

## Diversity
# MAO (most ambiguous and orthogonal)
# lambda
# diversity by clustering

import pandas as pd

from algorithms.active_learning import (
    ActiveLearner,
    RandomLearner,
    MarginSampler
)


def load_semeion(path, label_col=0):
    data = pd.read_csv(path, header=None, encoding='utf-8')

    return data.iloc[:, 1:].values, data.iloc[:, 0].values


if __name__ == '__main__':
    data = {
        'labeled': load_semeion('data/semeion_labeled.csv'),
        'unlabeled': load_semeion('data/semeion_unlabeled.csv'),
        'test': load_semeion('data/semeion_test.csv'),
    }

    kwargs = {
        'algorithms': {
            'RandomLearner': RandomLearner,
            'MarginSampler': MarginSampler,
        }
    }

    model = ActiveLearner(**kwargs)
    model.fit(**{'data': data})
