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
    MAODiversity,
    MAOLambda,
    MAOCluster,
)

from utils.plots import plot_comparison


def load_semeion(path, label_col=0):
    data = pd.read_csv(path, header=None, encoding='utf-8')

    return data.iloc[:, label_col+1:].values, data.iloc[:, label_col].values


if __name__ == '__main__':
    data = {
        'labeled': load_semeion('data/semeion_labeled.csv'),
        'unlabeled': load_semeion('data/semeion_unlabeled.csv'),
        'test': load_semeion('data/semeion_test.csv'),
    }

    kwargs = {
        'algorithms': {
            'Random': RandomLearner,
            'Margin Sampling': MarginSampling,
            'Multiclass Uncertainty': MulticlassUncertainty,
            'SignificanceSpaceConstruction': SignificanceSpaceConstruction,
            'nEQB': nEQB,
            'MAODiversity': MAODiversity,
            'MAOLambda': MAOLambda,
            'MAOCluster': MAOCluster,
        }
    }

    model = ActiveLearner(**kwargs)
    model.fit(**{'data': data})

    plot_comparison(model.scores)
