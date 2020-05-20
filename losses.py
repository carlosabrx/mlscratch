import numpy as np

''' MSE is used for regression models while cross entropy is for classification models'''

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError

    def loss_slope(self, y_true, y_pred):
        raise NotImplementedError

class MSE(Loss):
    def __init__(self):
        pass

    def square_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

class CrossEntropy(Loss):
    def __init__(self):
        pass
    def cel(self, y, p):
        return -(y * np.log(p) + (1 - y) * np.log(1-p))
