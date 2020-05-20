import numpy as np
from mlscratch.activation import Sigmoid, ReLu, TanH, SoftMax
from mlscratch.losses import MSE, CrossEntropy
import copy

class Layer:
    def base_shape(self, shape):
        self.input_shape = shape
    def layer_name(self):
        return self.__class__.name
    def param(self):
        return 0
    def fwd_pass(self, X, train):
        raise NotImplementedError
    def bwd_pass(self, gradients):
        raise NotImplementedError
    def output_shape(self):
        raise NotImplementedError

class Dense:
    def __init__(self, neurons, input_shape=None):
        self.input_shape = input_shape
        self.neurons = neurons
        self.layer_input = None
        self.trainable = True
        self.lr = 0.001
        self.W = None
        self.w0 = None

    def init(self, opt):
        limit = 1 / np.sqrt(self.input_shape[0]) #input shape of the rows
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.neurons)) #weights
        self.w0 = np.zeros((1, neurons)) #bias
        #optimizing weights
        self.W_opt = copy.copy(opt)
        self.w0_opt = copy.copy(opt)

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape) #len of sum of shapes

    def fwd_pass(self, X, train=True):
        self.input_layer = X
        return np.dot(X, self.W) + self.w0

    def bwd_pass(self, gradients):
        W = self.W #weights used during fwd_pass
        if self.trainable:
            #gradient w.r.t layer weights
            grad_w = self.layer_input.T.dot(gradients)
            grad_w0 = np.sum(gradients, axis=0, keepdims=True)

            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)
        gradients = gradients.dot(W.T)
        return gradients

    def output_shape(self):
        return (self.neurons, )
