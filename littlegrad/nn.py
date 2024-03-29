import numpy as np
from littlegrad.engine import Value

class Neuron:
    def __init__(self, w):
        self.w = np.random.random(w)
    def __call__(self, x):
        return np.dot(self.w, x)
