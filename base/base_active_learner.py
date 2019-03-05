SEED = 42

import numpy as np
np.random.seed(SEED)

from sklearn.svm import SVC


class BaseActiveLearner(object):
    ''' Base class for Active Learning objects
    '''

    def __init__(self):
        self.X_unlab, self.X_lab = None, None
        self.y_unlab, self.y_lab = None, None


    def init_labels(self, X_unlab, X_lab, y_unlab, y_lab):
        self.X_unlab, self.X_lab = X_unlab, X_lab
        self.y_unlab, self.y_lab = y_unlab, y_lab


    def update_labels(self, indices):
        ''' Auxiliary method for updating the labels, moving them from the pool
        to the training set
        '''
        self.X_lab = \
            np.concatenate((self.X_lab, self.X_unlab[indices, :]), axis=0)
        self.y_lab = \
            np.concatenate((self.y_lab, self.y_unlab[indices]), axis=0)

        self.X_unlab = np.delete(self.X_unlab, indices, axis=0)
        self.y_unlab = np.delete(self.y_unlab, indices, axis=0)


    def get_indices(self, n_points=1):
        ''' Placeholder method to be modified for each aproximation.

        This method will be different for each algorithm and will allow
        choosing samples from the pool heuristically.

        NOTE: use self.X_unlab
        '''
        raise NotImplementedError
