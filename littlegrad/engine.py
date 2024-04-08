import math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, children = (), _op = ''):
        self.data = data
        self.children = set(children) #forgetting the set() causes gradient descent to break (maybe because without set children would be tuple which would make .grad unchangable?)
        self.grad = 0
        self._backward = lambda: None
        self._op = _op #for jupyter notebook testing (draw_dot())
        self.momentum = 0  #gradient descent optimization attempt

    def data(self):
        return self.data
    
    def __repr__(self):
        return f"Value({self.data}, grad={self.grad})" #f-string
    
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
            other.grad += (self.data ** other.data) * math.log(max(abs(self.data), 1e-10)) * out.grad #assumes base is positive (otherwise function would be complex)
            #TODO: fix the above line? (how does tinygrad do?)
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
            self.grad += (other.data ** self.data) * math.log(abs(other.data)) * out.grad #assumes base is positive (otherwise function would be complex)

        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rtruediv__(self, other):
        return other * (self ** -1)
    
    def __lt__(self, other):
        other = other if (type(other) == Value) else Value(other)
        return self.data < other.data
    def __le__(self, other):
        other = other if (type(other) == Value) else Value(other)
        return self.data <= other.data
    def __gt__(self, other):
        other = other if (type(other) == Value) else Value(other)
        return self.data > other.data
    def __ge__(self, other):
        other = other if (type(other) == Value) else Value(other)
        return self.data >= other.data
    
    #can't have __eq__ because set() needs that to confirm that Value is hashable value (checks if hashes are identical?)
    
    def exp(self):
        return math.e**self
    
    def log(self):
        if self.data < 1e-100: self.data = 1e-100  #to avoid overflow/underflow errors
        out = Value(math.log(self.data), children = (self,), _op = 'log()')

        def _backward():
            self.grad += out.grad/self.data
            
        out._backward = _backward #NOTE: this used to be in the _backward() function and prevented nll loss from running
        return out
    
    def relu(self):
        out = Value(max([self.data, 0]), children = (self,), _op = "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out
    
    def backward(self):
        nodeList = []
        visited = set() #searching set is way faster than searching array for some reason
        def getNodes(val):
            if val not in visited:
                for child in val.children:
                    getNodes(child)
                nodeList.append(val)
                visited.add(val)
        getNodes(self)

        self.grad = 1.0
        for node in reversed(nodeList):
            node._backward()