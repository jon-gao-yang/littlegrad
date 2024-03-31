import torch
from littlegrad.engine import Value

a = -5
b = 3
c = 1

aVal = Value(a)
bVal = Value(b)
print("a:      | Passed == ", type(aVal) == Value)
print("b:      | Passed == ", type(bVal) == Value)
print("a.data: | Passed == ", aVal.data == a)
print("b.data: | Passed == ", bVal.data == b)
print("a+c:    | Passed == ", (aVal+c).data == a+c)
print("c+a:    | Passed == ", (c+aVal).data == c+a)
print("a-c:    | Passed == ", (aVal-c).data == a-c)
print("c-a:    | Passed == ", (c-aVal).data == c-a)
print("a*c:    | Passed == ", (aVal*c).data == a*c)
print("c*a:    | Passed == ", (c*aVal).data == c*a)
print("-a:     | Passed == ", (-aVal).data == -a)
print("a/c:    | Passed == ", (aVal/c).data == a/c)
print("c/a:    | Passed == ", (c/aVal).data == c/a)
print("a**c:   | Passed == ", (aVal**c).data == a**c)
print("c**a:   | Passed == ", (c**aVal).data == c**a)

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print("karpathy fwd pass:  | Passed == ", round(g.data, 4) == 24.7041) # prints 24.7041, the outcome of this forward pass

g.backward()
print("karpathy bckpass 1: | Passed == ", round(a.grad, 4) == 138.8338) # prints 138.8338, i.e. the numerical value of dg/da
print("karpathy bckpass 2: | Passed == ", round(b.grad, 4) == 645.5773) # # prints 645.5773, i.e. the numerical value of dg/db

# from karpathy/micrograd test_engine.py:
def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()
    print('yey 1')

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
    print('yey 2')

test_sanity_check()
test_more_ops()