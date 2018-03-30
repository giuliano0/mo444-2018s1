import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.cross_validation import train_test_split

####################################################################
# Gradient Descent
####################################################################

def gd(train_X, train_y, test_X, test_y, learning_rate=0.0001,
       iterations=1000, lr_dampening=1.0, reg=0):
    [m, n] = train_X.shape
    w = np.asarray(np.hstack((100, train_X.mean())))[:, np.newaxis]
    #w = np.random.randn(n + 1, 1)
    train_X = np.concatenate((np.ones((m, 1)), train_X), axis=1)

    train_y = train_y.as_matrix()

    if train_y.shape == (1, m):
        train_y = train_y.T
    if train_y.shape != (m, 1):
        print('train_y shape sould be m rows, 1 column')

    loss_history = np.zeros((iterations, 2)) # keeps track of MSE [train, test]
    #grad_history = np.ndarray(shape=(1,n+1)) # tracks the gradient itself (debugging)
    #w_history = w # tracks weights (debugging)

    for it in range(iterations):
        predictions = np.dot(train_X, w)
        loss = predictions - train_y
        loss_train = mean_squared_error(train_y, predictions)

        predictions = w[0] + np.dot(test_X, w[1:])
        loss_test = mean_squared_error(test_y, predictions)

        loss_history[it] = [loss_train, loss_test]

        grad = np.dot(train_X.T, loss) / m
        #grad_history = np.append(grad_history, grad.T, axis=0)

        if ((it + 1) % 100) == 0:
            print('it=%4d, loss=%.3f, 1000*lr=%.12f' % (it + 1,
                    loss_train, 1000 * learning_rate))

        if np.isnan(np.sum(grad)) or np.isinf(np.sum(grad)):
            print('NaN or Inf detected, stopping at it=' + str(it))
            break

        #              learning term             regularization term
        w = w - (learning_rate / (1+it)) * grad + reg * np.dot(w.ravel(), w.ravel())
        learning_rate = lr_dampening * learning_rate
        #w_history = np.append(w_history, w)

    return w.flatten(), loss_history#, grad_history #, w_history


#####################################################################
# lab
#####################################################################

from sklearn.linear_model import LinearRegression, SGDRegressor
from scipy import percentile

# ======================================
# Data loading (train and test as 2 DFs)
# ======================================
train_X = pd.read_csv('../data/train.csv')

# Merge test now, split again later
test_X = pd.read_csv('../data/test.csv')
test_y = pd.read_csv('../data/test_target.csv')

#test = pd.concat([test_X, test_y], axis=1)

# ===========================
# Outlier removal (split off)
# ===========================
cutoff = percentile(train_X['shares'], 90)

train_tail = train_X[train_X['shares'] >  cutoff]
train_X    = train_X[train_X['shares'] <= cutoff]

# ===========================
# (X, y) split for train data
# ===========================
train_y = pd.DataFrame(train_X['shares'], columns=['shares'])

train_X.drop('shares', axis=1, inplace=True)

# =================
# Feature selection
# =================
train_X.drop(['url', 'timedelta'], axis=1, inplace=True)

test_X.drop(['url', 'timedelta'], axis=1, inplace=True)


features_to_drop = [ ]
'''
    'weekday_is_monday',
    'weekday_is_tuesday',
    'weekday_is_wednesday',
    'weekday_is_thursday',
    'weekday_is_saturday',
    'weekday_is_sunday'
]
'''

train_X.drop(features_to_drop, axis=1, inplace=True)
test_X.drop(features_to_drop, axis=1, inplace=True)

# ============
# Data scaling
# ============
scaler = MinMaxScaler((0,10))

train_X = pd.DataFrame(scaler.fit_transform(train_X), columns=train_X.columns)
test_X  = pd.DataFrame(scaler.transform(test_X),      columns=test_X.columns)

# Data is scaled now let's train

# ==================================
# Training with my GD implementation
# ==================================

# Repeating k-folds CV
#rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
#for train, test in rkf.split(X):

w, losses = gd(train_X, train_y, test_X, test_y, iterations=1000,
               learning_rate=1e-3, lr_dampening=0.9999)


my_preds = w[0] + np.dot(test_X, w[1:])

# ===========================================
# Auxiliar regressor sklearn.LinearRegression
# ===========================================
lg_model = LinearRegression()
lg_model.fit(train_X, train_y)
lg_preds = lg_model.predict(test_X).ravel()

all_preds_y = np.vstack((test_y.as_matrix().ravel(), my_preds, lg_preds)).T

print('LR: my_preds_loss / sk_preds_loss = %.5f' % (
        mean_squared_error(my_preds, test_y) /
        mean_squared_error(lg_preds, test_y)))

# ========================================
# Auxiliary regressor sklearn.SGDRegressor
# ========================================
sgd_model = SGDRegressor(alpha=1e-4, max_iter=1, tol=1e-4, eta0=1e-4,
                         learning_rate='invscaling', warm_start=True)
sgd_loss = np.zeros((100, 1))

for it in range(100):
    sgd_model.fit(train_X, train_y)

    if ((it + 1) % 10 == 0):
        sgd_preds = sgd_model.predict(test_X)
        sgd_loss[it] = mean_squared_error(sgd_preds, test_y)

    if ((it + 1) % 100 == 0):
        print('iteration ' + str(it + 1))

sgd_preds = sgd_model.predict(test_X)#.ravel()

all_preds_y = np.vstack((all_preds_y.T, sgd_preds)).T
all_preds = pd.DataFrame(all_preds_y, columns=['truth', 'my gd', 'sk.lr', 'sk.sgd'])

print('SGD: my_preds_loss / sk_preds_loss = %.5f' % (
        mean_squared_error(my_preds, test_y) /
        mean_squared_error(sgd_preds, test_y)))

print('finished')
#print(w)

plt.plot(losses[:20, 0], label='train loss')
plt.plot(losses[:20, 1], label='test loss')
#plt.ylim([1.0e8, 2.4e8])
plt.legend()
plt.show()

print('Absolute difference of means (train and test loss): %.4f * 10e7' %
      (np.abs(np.mean(losses[:, 0]) - np.mean(losses[:, 1])) / 10e7))

# What is the distribution of the absolute residuals for my model?
#plt.hist(all_preds['truth'] - all_preds['my gd'], bins=30)

# Let's take a closer look at log(residuals)
residuals = np.abs(all_preds['my gd'] - all_preds['truth'])
plt.plot(np.random.choice(np.log(residuals),
                          size=int(1.0 * len(all_preds)), replace=False),
        '.', markersize=1, color='slateblue')
plt.ylim([-2, 16])


