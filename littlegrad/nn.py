import random
from littlegrad.engine import Value

class Module:
    def parameters(self):
        return []
    
    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0

class Neuron(Module):
    def __init__(self, wNum, lin = False):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(wNum)]
        self.b = Value(0.0)
        self.lin = lin

    def parameters(self):
        return self.w + [self.b]
    
    def __call__(self, x):
        act = sum([xi*wi for (xi, wi) in zip(x, self.w)], self.b)
        return act if self.lin else act.relu()
    
    def __repr__(self):
        return f"{'Linear' if self.lin else 'ReLU'}_Neuron({len(self.w)})" #f-string

class Layer(Module):
    def __init__(self, wNum, neuronNum, **kwargs):
        self.layer = [Neuron(wNum, **kwargs) for _ in range(neuronNum)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.layer]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        #return [parameter for parameter in neuron.parameters() for neuron in self.layer]
        return [parameter for neuron in self.layer for parameter in neuron.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(neuron) for neuron in self.layer)}]"
    
class MLP(Module):
    def __init__(self, xNum, neuronList):
        sizes = [xNum] + neuronList
        #self.mlp = [Layer(xNum, neuronNum) for neuronNum in neuronList]
        self.mlp = [Layer(sizes[i], sizes[i+1], lin = (i == (len(neuronList)-1))) for i in range(len(neuronList))]

    def __call__(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x
    
    def parameters(self):
        return [parameter for layer in self.mlp for parameter in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.mlp)}]"