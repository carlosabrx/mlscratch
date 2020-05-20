import numpy as np

class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    def slope(self, x):
        return (1 + np.exp(-x) + (x * np.exp(-x)))/(1 + np.exp(-x))**2

class ReLu():
    def __call__(self, x):
        if x >= 0:
            return max(0,x)
    def slope(self, x):
        return np.where(x >= 0, 1, 0)

class TanH():
    def __call__(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    def slope(self, x):
        return (1 - (self.__call__(x))**2)

class SoftMax():
    def __call__(self, x):
        ex = np.exp(x - np.max(x))
        return ex/ex.sum()
    def slope(self, x):
        m = self.__call__(x)
        return m * (1 - m)
