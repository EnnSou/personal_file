class Value():
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0
        self._backwoard = lambda:None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value((self.data + other.data), (self, other), '+')
        
        def _backwoard():
            self.grad += out.grad * 1
            print(self.grad)
            other.grad += out.grad * 1
            print(other.grad)
        out._backwoard = _backwoard

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value((self.data * other.data), (self, other), '*')

        def _backwoard():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backwoard = _backwoard

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value((self.data**other), (self,), f'**{other}')

        def _backwoard():
            self.grad += out.grad * (other * self.data**(other - 1))
        out._backwoard = _backwoard

        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backwoard():
            self.grad += (out.data > 0) * out.grad

        out._backwoard = _backwoard

        return out

    
    def backwoard(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backwoard()

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
    
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, op={self._op})"
    