import torch
from littlegrad.engine import Value
from littlegrad.nn import *
import matplotlib.pyplot as plt
import numpy as np
import csv

def my_test():
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
    print()

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
    print('Karpathy #1: Passed == True')

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
    print('Karpathy #2: Passed == True')
    print()

def mlp_test():
    n = MLP(3, [4, 4, 1])
    xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets

    for k in range(100):

        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        
        # backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()
        
        # update
        for p in n.parameters():
            p.data += -0.01 * p.grad
        
        print(k, loss.data)

    print(ypred)
    print()

#based on a practice assignment from Andrew Ng's "Advanced Learning Algorithms" Coursera course
def plot_kaggle_data(X, y, model, predict=False):
    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((28,28))
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        yhat = None
        # Predict using the Neural Network
        if predict:
            
            inputs = [list(map(Value, X[random_index]))] # pick random image and convert pixel ints to Value, 
            prediction = list(map(model, inputs))[0]        # then plug pixel Values into model for fwd pass ([0] because list(map) returns a list of lists which breaks softmax)             
            prediction_p = softmax(prediction)
            #print()
            #print('PREDICTION_P', prediction_p)
            yhat = np.argmax([prediction.data for prediction in prediction_p]) #TODO: fix this? (add max/argmax to Value class?)
            #print('YHAT:', yhat)        
        
        # Display the label above the image
        ax.set_title(f"{int(y[random_index])},{yhat}",fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=14)
    plt.show()

#from karpathy's micrograd_exercises.ipynb
def softmax(logits):
  #print('LOGITS:', logits)
  #counts = [logit.exp() for logit in logits]
  counts = [(logit/1000).exp() for logit in logits] #TODO: fix this?
  denominator = sum(counts)
  out = [c / denominator for c in counts]
  #print('SOFTMAX:', out)
  return out

#modified from karpathy's demo.ipynb
def loss(X, y, model, batch_size=None):
    
    # inline DataLoader :)
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]
    # forward the model to get scores
    scores = list(map(model, inputs))
    # negative log likelihood loss
    losses = [-softmax(scorei)[yi.item()].log() for (yi,scorei) in zip(yb, scores)] #can't take softmax of a list of lists
    #print("LOSSES:", losses)
    data_loss = sum(losses) * (1.0 / len(losses)) #cost = average loss
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p.data**2 for p in model.parameters())) #p.data to avoid involving p.grad
    total_loss = data_loss + reg_loss
    
    # also get accuracy
    #accuracy = [(yi.item()) == (np.argmax(scorei[yi.item()]).item()) for (yi, scorei) in zip(yb, scores)] 
    #return total_loss, sum(accuracy) / len(accuracy)
    return total_loss, 1 #TODO: fix accruacy and use total_loss (total_loss causes max recursion depth error for get_child?)

def kaggle_training():
    X = np.empty((42000, 28*28), dtype = int)
    y = np.empty(42000, dtype = int)
    with open('digit-recognizer/train.csv', newline='\n') as csvfile:
        digitreader = csv.reader(csvfile, delimiter=',')
        for row in digitreader:
            if digitreader.line_num != 1: #line_num starts at 1, not 0
                y[digitreader.line_num-2] = int(row[0])
                X[digitreader.line_num-2] = [int(char) for char in row[1:]]

    # initialize a model 
    #model = MLP(28*28, [25, 15, 10])
    model = MLP(28*28, [25, 15, 10])
    print()
    print(model)
    print()
    print("NUMBER OF PARAMTERS:", len(model.parameters()))
    print()

    # optimization
    k_range = 24
    for k in range(k_range):
        
        # forward
        #total_loss, acc = loss(X, y, model, batch_size = 32)
        total_loss, acc = loss(X, y, model, batch_size = 16)
        
        # backward
        model.zero_grad()
        total_loss.backward()
        
        # update (sgd)
        #learning_rate = 1.0 - 0.9*k/100
        learning_rate = 0.1 - 0.09*k/k_range #test
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        
        if k % 1 == 0:
            print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

    plot_kaggle_data(X, y, model, predict = True)

#############################################################################################

#my_test()
#test_sanity_check()
#test_more_ops()
#mlp_test()
    
kaggle_training()