## Diversity
# lambda
# diversity by clustering

import pandas as pd

from algorithms.active_learning import (
    ActiveLearner,
    RandomLearner,
    MarginSampling,
    MulticlassUncertainty,
    SignificanceSpaceConstruction,
    nEQB,
)
from algorithms.diversity import MAODiversity, MAOLambda, MAOCluster

from utils.plots import plot_comparison


def load_semeion(path, label_col=0):
    data = pd.read_csv(path, header=None, encoding='utf-8')

    return data.iloc[:, label_col+1:].values, data.iloc[:, label_col].values


AL_ALGORITHMS = [
    RandomLearner,
    MarginSampling,
    MulticlassUncertainty,
    SignificanceSpaceConstruction,
    # nEQB,
]

AL_DIVERSITY = [MAODiversity, MAOLambda, MAOCluster]


if __name__ == '__main__':
    data = {
        'labeled': load_semeion('data/semeion_labeled.csv'),
        'unlabeled': load_semeion('data/semeion_unlabeled.csv'),
        'test': load_semeion('data/semeion_test.csv'),
    }

    algorithms = {
        '{} - {}'.format(alg_name.__name__, alg_criterion.__name__): \
            (alg_name, alg_criterion)
        for alg_name in AL_ALGORITHMS
        for alg_criterion in AL_DIVERSITY
    }

    kwargs = {'algorithms': algorithms}

    model = ActiveLearner(**kwargs)
    model.fit(**{'data': data})

    plot_comparison(model.scores)
