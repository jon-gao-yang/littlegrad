
class Value:
    def __init__(self, data):
        self.data = data

    def data(self):
        return self.data
    
    def __add__(self, other):
        other = other if (type(other) == Value) else Value(other)
        return Value(self.data + other.data)
        
    def __mul__(self, other):
        other = other if (type(other) == Value) else Value(other)
        return Value(self.data * other.data)
        
    def __pow__(self, other):
        other = other if (type(other) == Value) else Value(other)
        return Value(self.data ** other.data)
        
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other ** -1)
        
    def __neg__(self):
        return Value(self.data * -1)
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rpow__(self, other):
        other = other if (type(other) == Value) else Value(other)
        return Value(other.data ** self.data)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rtruediv__(self, other):
        return other * (self ** -1)
    
    def relu(self):
        return Value(max([self.data, 0]))