import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def gd(train_X, train_y, test_X, test_y, learning_rate=0.0001, iterations=1000, lr_dampening=1.0):
    [m, n] = train_X.shape
    w = np.asarray(np.hstack((100, trainX.mean())))[:, np.newaxis]
    #w = np.random.randn(n + 1, 1)
    train_X = np.concatenate((np.ones((m, 1)), train_X), axis=1)
    
    loss_history = np.zeros((iterations, 2)) # keeps track of MSE [train, test]
    grad_history = np.ndarray(shape=(1,n+1)) # tracks the gradient itself (debugging)
    w_history = w # tracks weights (debugging)
    
    for it in range(iterations):
        predictions = np.dot(train_X, w)
        loss = predictions - train_y
        mse_train = mean_squared_error(train_y, predictions)
        
        predictions = w[0] + np.dot(test_X, w[1:])
        mse_test = mean_squared_error(test_y, predictions)
        
        loss_history[it] = [mse_train, mse_test]
        
        grad = np.dot(train_X.T, loss) / m
        grad_history = np.append(grad_history, grad.T, axis=0)
        
        if ((it + 1) % 100) == 0:
            print('it=%4d, mse=%.3f, 1000*lr=%.12f' % (it + 1, mse_train, 1000 * learning_rate))
        
        if np.isnan(np.sum(grad)) or np.isinf(np.sum(grad)):
            print('NaN or Inf detected, stopping at it=' + str(it))
            break
        
        w = w - (learning_rate / (1+it)) * grad
        learning_rate = lr_dampening * learning_rate
        w_history = np.append(w_history, w)
    
    return w.flatten(), loss_history, grad_history, w_history


# lab

import pandas as pd
from sklearn.linear_model import LinearRegression

trainX = pd.read_csv('train.csv')
trainy = (trainX['shares'])[:, np.newaxis]

trainX.drop('shares', axis=1, inplace=True)
trainX.drop('url', axis=1, inplace=True)
trainX.drop('timedelta', axis=1, inplace=True)

trainX = trainX.astype(np.float32)
trainy = trainy.astype(np.float32)

scaler = MinMaxScaler((0,10))
trainX = pd.DataFrame(scaler.fit_transform(trainX), columns=trainX.columns)


testX = pd.read_csv('test.csv')
testy = pd.read_csv('test_target.csv')

testX.drop('url', axis=1, inplace=True)
testX.drop('timedelta', axis=1, inplace=True)

testX = testX.astype(np.float32)
testy = testy.astype(np.float32)

testX = pd.DataFrame(scaler.transform(testX), columns=testX.columns)


w, losses, grads, ws = gd(trainX, trainy, testX, testy, iterations=10000,
                          learning_rate=1e-3, lr_dampening=0.9999)


my_preds = w[0] + np.dot(testX, w[1:])

model = LinearRegression()
model.fit(trainX, trainy)
sk_preds = model.predict(testX).ravel()

all_preds_y = np.vstack((testy.as_matrix().ravel(), my_preds, sk_preds)).T

print('my_preds_mse / sk_preds_mse = %.5f' % (mean_squared_error(my_preds, testy) / mean_squared_error(sk_preds, testy)))

print('finished')
#print(w)

plt.plot(losses[:, 0])
plt.plot(losses[:, 1])
plt.ylim([1.0e8, 2.4e8])
plt.show()
