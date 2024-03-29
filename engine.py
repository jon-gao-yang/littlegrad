import numpy as np

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, children = (), _op = ''):
        self.data = data
        self.children = children
        self.grad = 0
        self._backward = lambda: None
        self._op = _op #for jupyter notebook testing (draw_dot())

    def data(self):
        return self.data
    
    def __add__(self, other):
        other = other if (type(other) == Value) else Value(other)
        out = Value(self.data + other.data, children = (self, other), _op = '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
        
    def __mul__(self, other):
        other = other if (type(other) == Value) else Value(other)
        out = Value(self.data * other.data, children = (self, other), _op = '*')

        def _backward():
            self.grad += (other.data * out.grad)
            other.grad += (self.data * out.grad)

        out._backward = _backward
        return out
        
    def __pow__(self, other): 
        other = other if (type(other) == Value) else Value(other)
        out = Value(self.data ** other.data, children = (self, other), _op = '**')

        def _backward():
            self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
            other.grad += (self.data ** other.data) * np.log(abs(self.data)) * out.grad #assumes base is positive (otherwise function would be complex)

        out._backward = _backward
        return out
        
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other ** -1)
        
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rpow__(self, other): 
        other = other if (type(other) == Value) else Value(other)
        out = Value(other.data ** self.data, children = (other, self), _op = "**")

        def _backward():
            other.grad += (self.data * (other.data ** (self.data - 1))) * out.grad
            self.grad += (other.data ** self.data) * np.log(abs(other.data)) * out.grad #assumes base is positive (otherwise function would be complex)

        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rtruediv__(self, other):
        return other * (self ** -1)
    
    def relu(self):
        out = Value(max([self.data, 0]), children = (self,), _op = "ReLU")

        def _backward():
            self.grad += int(self.data > 0) * out.grad

        out._backward = _backward
        return out
    
    def backward(self):
        nodeList = []
        def getNodes(val):
            if val not in nodeList:
                for child in val.children:
                    getNodes(child)
                nodeList.append(val)
        getNodes(self)

        self.grad = 1.0
        for node in reversed(nodeList):
            node._backward()