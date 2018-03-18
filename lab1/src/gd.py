import numpy as np
from matplotlib import pyplot as plt

def gd(train_X, train_y, learning_rate=0.0001, iterations=1000, lr_dampening=1.0):
    [m, n] = train_X.shape
    w = np.random.randn(n + 1, 1)
    train_X = np.concatenate((np.ones((m, 1)), train_X), axis=1)
    
    loss_history = np.zeros(iterations) # keeps track of MSE
    grad_history = np.ndarray(shape=(1,n+1)) # tracks the gradient itself (debugging)
    w_history = w # tracks weights (debugging)
    
    for it in range(iterations):
        predictions = np.dot(train_X, w)
        loss = predictions - train_y
        mse = np.dot(loss.T, loss).flatten()[0] / m
        
        loss_history[it] = mse
        
        grad = np.dot(train_X.T, loss) / m
        grad_history = np.append(grad_history, grad.T, axis=0)
        
        if np.isnan(np.sum(grad)) or np.isinf(np.sum(grad)):
            print('NaN or Inf detected, stopping at it=' + str(it))
            break
        
        learning_rate = lr_dampening * learning_rate
        w = w - learning_rate * grad
        w_history = np.append(w_history, w)
    
    return w.flatten()


# lab

import pandas as pd

trainX = pd.read_csv('train.csv')
#trainy = pd.read_csv('test.csv')

#columns = train_data.columns.values
#select_features = dict(zip(columns, np.ones(len(columns), dtype=int)))

trainy = (trainX['shares'])[:, np.newaxis]

trainX.drop('shares', axis=1, inplace=True)
trainX.drop('url', axis=1, inplace=True)

trainX = trainX.astype(np.float32)
trainy = trainy.astype(np.float32)

w = gd(trainX, trainy, iterations=10000, learning_rate=1e-9, lr_dampening=0.9)

print('finished')
print(w)
