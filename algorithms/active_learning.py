SEED = 42

import numpy as np
np.random.seed(SEED)

import pandas as pd

from scipy.spatial.distance import pdist

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from collections import defaultdict

from base.base_active_learner import BaseActiveLearner


class ActiveLearner(object):
    '''Implementation of Active Learning algorithm and its random counterpart
    '''

    def __init__(self, **kwargs):
        self.test_size = self.lab_size = 0.33

        self.n_queries = 100
        self.n_update_points = 10

        self.estimator = SVC(C=100, gamma=1, decision_function_shape='ovr')
        self.algorithms = dict()

        # TODO: add scoring function

        self.seed = SEED

        # Update class attributes with keyword arguments (hyperparams)
        self.__dict__.update(kwargs)

        # Create dataframe to store scores for every estimator
        # self.scores = pd.DataFrame({'Iteration': range(1, self.n_queries + 1)})
        self.scores = pd.DataFrame()


    def _load_data(self, X=None, y=None, data=None):
        '''Auxiliary method for allowing either the use of whole raw datasets
        or of already splitted ones
        '''
        if X and y:
            # Create train/test in main
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.seed
            )

            # Create lab/unlab split with train
            X_unlab, X_lab, y_unlab, y_lab = train_test_split(
                X_train, y_train, test_size=self.lab_size,
                stratify=y_train, random_state=self.seed
            )

            return X_unlab, X_lab, y_unlab, y_lab, X_test, y_test

        elif data:
            X_lab, y_lab = data['labeled']
            X_unlab, y_unlab = data['unlabeled']
            X_test, y_test = data['test']

            return X_unlab, X_lab, y_unlab, y_lab, X_test, y_test


    def fit(self, **kwargs):
        X_unlab, X_lab, y_unlab, y_lab, X_test, y_test = \
            self._load_data(**kwargs)

        # Initalize labels in all algorithms (they must start with the same)
        # and base estimator
        for est_name, est_class in self.algorithms.items():
            est_class = est_class(self.estimator)
            est_class.init_labels(X_unlab, X_lab, y_unlab, y_lab)

            self.algorithms[est_name] = est_class

        for alg_name, alg_class in self.algorithms.items():
            alg_scores = []

            for i in range(self.n_queries):
                # 1. Fit estimator
                alg_class.estimator.fit(alg_class.X_lab, alg_class.y_lab)

                # 2. Obtain score on X_test, y_test
                alg_scores.append(alg_class.estimator.score(X_test, y_test))

                # 3. Choose some random samples from X_unlab, y_unlab
                indices = alg_class.get_indices(n_points=self.n_update_points)

                # 4. Update labels, moving them from pool to train
                alg_class.update_labels(indices)

            self.scores[alg_name] = alg_scores



class RandomLearner(BaseActiveLearner):

    def __init__(self, estimator):
        super(RandomLearner, self).__init__()

        self.estimator = estimator


    def get_indices(self, n_points=1):
        indices = np.random.permutation(self.X_unlab.shape[0])
        indices = indices[0:n_points]

        return indices



class MarginSampler(BaseActiveLearner):

    def __init__(self, estimator):
        super(MarginSampler, self).__init__()

        self.estimator = estimator


    def get_indices(self, n_points=1):
        # TODO: change for the proper algorithm
        indices = np.random.permutation(self.X_unlab.shape[0])
        indices = indices[0:n_points]

        return indices
