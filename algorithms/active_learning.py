SEED = 42

import numpy as np
np.random.seed(SEED)

import pandas as pd

from scipy.spatial.distance import pdist

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from collections import defaultdict

from copy import copy

from tqdm import tqdm

from base.base_active_learner import BaseActiveLearner


class ActiveLearner(object):
    '''Implementation of Active Learning algorithm and its random counterpart
    '''

    def __init__(self, **kwargs):
        self.test_size = self.lab_size = 0.33

        self.n_queries = 50
        self.n_update_points = 10
        self.algorithms = dict()

        # TODO: rearrange self.estimator in order to add gamma adjusted to data
        # TODO: add scoring function

        self.seed = SEED

        # Update class attributes with keyword arguments (hyperparams)
        self.__dict__.update(kwargs)

        # Create dataframe to store scores for every estimator
        self.scores = pd.DataFrame()


    def _load_data(self, X=None, y=None, data=None):
        ''' Auxiliary method for allowing either the use of whole raw datasets
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

        # TODO: rearrange in __init__
        # Update estimator with gamma adjusted to data
        sigma = np.mean(pdist(X_lab))
        gamma = 1 / (2 * sigma * sigma)
        base_estimator = SVC(C=100, gamma=gamma, decision_function_shape='ovr')

        for alg_name, alg_class in self.algorithms.items():
            # Initalize labels in all algorithms (they must start with the
            # same labels) and base estimator
            alg_class = alg_class(base_estimator)
            alg_class.init_labels(X_unlab, X_lab, y_unlab, y_lab)
            self.algorithms[alg_name] = alg_class

            print('[+] Algorithm: {}'.format(alg_name))

            alg_scores = []
            for _ in tqdm(range(self.n_queries)):
                # 1. Fit estimator
                alg_class.estimator.fit(alg_class.X_lab, alg_class.y_lab)

                # 2. Obtain score on X_test, y_test
                alg_scores.append(alg_class.estimator.score(X_test, y_test))

                # 3. Choose samples from X_unlab, y_unlab
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
        indices = indices[:n_points]

        return indices



class MarginSampling(BaseActiveLearner):

    def __init__(self, estimator):
        super(MarginSampling, self).__init__()

        self.estimator = estimator


    def get_indices(self, n_points=1):
        # Use heuristic to choose samples from pool
        dist = np.abs(self.estimator.decision_function(self.X_unlab))

        # Sort inside each sample from lowest to most probable classes
        dist = np.sort(dist, axis=1)

        # Sort along all samples to get the most uncertain samples, this is,
        # samples with lowest distance to hyperplane for its most probable class
        indices = np.argsort(dist[:, -1])

        return indices[:n_points]



class MulticlassUncertainty(BaseActiveLearner):

    def __init__(self, estimator):
        super(MulticlassUncertainty, self).__init__()

        self.estimator = estimator


    def get_indices(self, n_points=1):
        # Use heuristic to choose samples from pool
        dist = np.abs(self.estimator.decision_function(self.X_unlab))

        # Sort inside each sample from lowest to most probable classes
        dist = np.sort(dist, axis=1)

        # Sort along all samples to get the most uncertain samples, this is,
        # samples with lowest difference between the most probable classes
        indices = np.argsort(dist[:, -1] - dist[:, -2])

        return indices[:n_points]



class SignificanceSpaceConstruction(BaseActiveLearner):

    def __init__(self, estimator):
        super(SignificanceSpaceConstruction, self).__init__()

        self.estimator = estimator

        # Create second estimator, which will classify support vectors against
        # the rest of training samples, yielding samples that are likely to
        # become support vectors
        self.sv_estimator = copy(estimator)


    def get_indices(self, n_points=1):
        # Use heuristic to choose samples from pool
        X_train = self.X_lab
        y_train = np.zeros(self.y_lab.shape)

        # Update support vectors already found by the classifier
        y_train[self.estimator.support_] = 1

        if len(np.unique(y_train)) == 1:
            indices = np.random.permutation(self.X_unlab.shape[0])

        else:
            self.sv_estimator.fit(X_train, y_train)
            possible_SVs = self.sv_estimator.predict(self.X_unlab)
            indices = np.arange(self.X_unlab.shape[0])[possible_SVs == 1]
            indices = np.random.permutation(indices)

        return indices[:n_points]
