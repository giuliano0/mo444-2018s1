import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.model_selection import KFold, RepeatedKFold
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
    residual_history = np.zeros((iterations, 2)) # keeps track of MAE [train, test]
    #grad_history = np.ndarray(shape=(1,n+1)) # tracks the gradient itself (debugging)
    #w_history = w # tracks weights (debugging)

    for it in range(iterations):
        predictions = np.dot(train_X, w)
        loss = predictions - train_y
        loss_train = mse(train_y, predictions)
        residual_train = mae(train_y, predictions)

        predictions = w[0] + np.dot(test_X, w[1:])
        loss_test = mse(test_y, predictions)
        residual_test = mae(test_y, predictions)

        loss_history[it] = [loss_train, loss_test]
        residual_history[it] = [residual_train, residual_test]

        grad = np.dot(train_X.T, loss) / m
        #grad_history = np.append(grad_history, grad.T, axis=0)

        if ((it + 1) % 100) == 0:
            print('it=%4d, loss=%.3f, residual=%.3f 1000*lr=%.12f' % (it + 1,
                    loss_train, residual_train, 1000 * learning_rate))

        if np.isnan(np.sum(grad)) or np.isinf(np.sum(grad)):
            print('NaN or Inf detected, stopping at it=' + str(it))
            break

        #                learning term                 regularization term
        w = w - (learning_rate / (1+it)) * grad + reg * np.dot(w.ravel(), w.ravel())
        learning_rate = lr_dampening * learning_rate
        #w_history = np.append(w_history, w)
        
        if np.isnan(np.sum(w)) or np.isinf(np.sum(w)):
            print('NaN or Inf detected after w update, stopping at it=' + str(it))
            break

    return w.flatten(), loss_history, residual_history#, grad_history #, w_history


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
if True:
    cutoff = percentile(train_X['shares'], 95)
    
    train_tail = train_X[train_X['shares'] >  cutoff]
    train_X    = train_X[train_X['shares'] <= cutoff]


# ======================================
# Outlier removal (feature distribution)
# ======================================
if True:
    remove_outliers = [
        'n_tokens_content',
        'n_unique_tokens',
        'n_non_stop_words',
        'n_non_stop_unique_tokens',
        'num_hrefs',
        'num_self_hrefs',
        'num_imgs',
        'num_videos',
        'average_token_length',
        'kw_min_min',
        'kw_max_min',
        'kw_avg_min',
        #'kw_max_max',
        'kw_max_avg',
        'kw_avg_avg',
        'self_reference_min_shares',
        'self_reference_max_shares',
        'self_reference_avg_sharess',
    ]
    
    feature_outliers = pd.DataFrame(columns=train_X.columns)
    
    for col in remove_outliers:
        p = percentile(train_X[col], 99)
        
        feature_outliers = feature_outliers.append(train_X[train_X[col] > p])
        train_X = train_X[train_X[col] <= p]
    
    print(train_X.shape)
    print(feature_outliers.shape)


# =================
# Feature selection
# =================
train_X.drop(['url', 'timedelta'], axis=1, inplace=True)
test_X.drop(['url', 'timedelta'], axis=1, inplace=True)

features_to_drop = []

if False:
    features_to_drop = features_to_drop + [
        'weekday_is_monday',
        'weekday_is_tuesday',
        'weekday_is_wednesday',
        'weekday_is_thursday',
        'weekday_is_saturday',
        'weekday_is_sunday'
    ]
    
if False:
    features_to_drop = features_to_drop + [
        'kw_min_min',
        'kw_max_min',
        'kw_min_max',
        'kw_max_max',
        'kw_min_avg',
        'kw_max_avg'
    ]
    
if False:
    features_to_drop = features_to_drop + [
        'LDA_00',
        'LDA_01',
        'LDA_02',
        'LDA_03',
        'LDA_04'
    ]
    
if False:
    features_to_drop = features_to_drop + [
        'min_positive_polarity',
        'min_positive_polarity',
        'min_negative_polarity',
        'max_negative_polarity'
    ]

train_X.drop(features_to_drop, axis=1, inplace=True)
test_X.drop(features_to_drop, axis=1, inplace=True)


# ===========================
# (X, y) split for train data
# ===========================
train_y = pd.DataFrame(train_X['shares'], columns=['shares'])

train_X.drop('shares', axis=1, inplace=True)


# ===========================
# Increasing model complexity
# ===========================
if False:
    max_power = 2
    new_cols = pd.DataFrame()
    bin_features = []
    
    for col in train_X.columns:
        if len(np.unique(train_X[col])) == 2:
            bin_features.append(col)
    
    for col in train_X.columns.drop(bin_features):
        for p in list(range(2, max_power+1)):
            train_X[col + '_' + str(p)] = np.power(train_X[col], p)
            test_X[col + '_' + str(p)] = np.power(test_X[col], p)


# ======================================
# Data scaling (must be after X/y split)
# ======================================
scaler = MinMaxScaler((0,1))

train_X = pd.DataFrame(scaler.fit_transform(train_X), columns=train_X.columns)
test_X  = pd.DataFrame(scaler.transform(test_X),      columns=test_X.columns)


# ==================================
# Training with my GD implementation
# ==================================

# Repeating k-folds CV
#rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
#for train, test in rkf.split(X):

its = 11000 # no extra features
#its = 6500 # with extra features
#its = 8000 # with all outliers
folds = 5

kf = KFold(n_splits=folds, shuffle=True)
mean_losses = np.zeros((its, 2))
mean_residuals = np.zeros((its, 2))
xval_losses_residuals = np.zeros((folds, 2))

train_X.reset_index(drop=True)
train_y.reset_index(drop=True)
test_X.reset_index(drop=True)
test_y.reset_index(drop=True)

fold_it = 0

for train_idx, test_idx in kf.split(train_X):
    print('Starting k-fold iteration...')
    
    w, losses, residuals = gd(train_X.iloc[train_idx], train_y.iloc[train_idx],
                              train_X.iloc[test_idx], train_y.iloc[test_idx],
                              iterations=its, learning_rate=1e-3,
                              lr_dampening=0.999, reg=1e-7)
    
    mean_losses = mean_losses + (1/5) * losses
    mean_residuals = mean_residuals + (1/5) * residuals
    
    xval_preds = w[0] + np.dot(test_X, w[1:])
    
    xval_losses_residuals[fold_it] = [mse(test_y, xval_preds), mae(test_y, xval_preds)]
    
    fold_it = fold_it + 1

print('Cross-validation (a) loss: %.2f +- %.2f (b) residual: %.2f +- %.2f' % (
        np.mean(xval_losses_residuals[:, 0]), np.std(xval_losses_residuals[:, 0]),
        np.mean(xval_losses_residuals[:, 1]), np.std(xval_losses_residuals[:, 1]),))


# Final model
w, final_losses, final_residuals = gd(train_X, train_y, test_X, test_y,
                                      iterations=its, learning_rate=1e-3,
                                      lr_dampening=0.999, reg=1e-7)


my_preds = w[0] + np.dot(test_X, w[1:])


# ===========================================
# Auxiliar regressor sklearn.LinearRegression
# ===========================================
'''
lg_model = LinearRegression()
lg_model.fit(train_X, train_y)
lg_preds = lg_model.predict(test_X).ravel()

all_preds_y = np.vstack((test_y.as_matrix().ravel(), my_preds, lg_preds)).T

print('LR: my_preds_loss / sk_preds_loss = %.5f' % (
        mse(my_preds, test_y) /
        mse(lg_preds, test_y)))
'''

# ========================================
# Auxiliar regressor sklearn.SGDRegressor
# ========================================
'''
sgd_model = SGDRegressor(alpha=1e-4, max_iter=1, tol=1e-4, eta0=1e-4,
                         learning_rate='invscaling', warm_start=True)
sgd_loss = np.zeros((100, 1))

for it in range(100):
    sgd_model.fit(train_X, train_y)

    if ((it + 1) % 10 == 0):
        sgd_preds = sgd_model.predict(test_X)
        sgd_loss[it] = mse(sgd_preds, test_y)

    if ((it + 1) % 100 == 0):
        print('iteration ' + str(it + 1))

sgd_preds = sgd_model.predict(test_X)#.ravel()

all_preds_y = np.vstack((all_preds_y.T, sgd_preds)).T
all_preds = pd.DataFrame(all_preds_y, columns=['truth', 'my gd', 'sk.lr', 'sk.sgd'])

print('SGD: my_preds_loss / sk_preds_loss = %.5f' % (
        mse(my_preds, test_y) /
        mse(sgd_preds, test_y)))

print('finished')
'''


plt.plot(np.sqrt(mean_losses[:, 0]), label='train loss')
plt.plot(np.sqrt(mean_losses[:, 1]), label='validation loss')
plt.title('MRSE during cross-validation')
plt.legend()
plt.show()

plt.plot(mean_residuals[:, 0], label='train residuals', alpha=0.9)
plt.plot(mean_residuals[:, 1], label='validation residuals', alpha=0.5)
#plt.ylim([1.0e8, 2.4e8])
plt.title('MAE during cross-validation')
plt.legend()
plt.show()


plt.plot(np.sqrt(losses[:, 0]), label='train loss')
plt.plot(np.sqrt(losses[:, 1]), label='test loss')
#plt.ylim([1.0e8, 2.4e8])
plt.title('Final model RMSE during training')
plt.legend()
plt.show()

plt.plot(losses[:20, 0], label='train residuals')
plt.plot(losses[:20, 1], label='test residuals')
#plt.ylim([1.0e8, 2.4e8])
plt.title('Final model MAE during training')
plt.legend()
plt.show()


print('Absolute difference of means (train and test loss): %.4f * 1' %
      (np.abs(np.mean(losses[:, 0]) - np.mean(losses[:, 1])) / 1))

# ==================
# Direct calculation
# ==================
from numpy.linalg import pinv

the_inverse = pinv(np.dot(train_X.T, train_X))
np.dot(the_inverse, train_X.T)
left_mat = np.dot(the_inverse, train_X.T)
theta = np.dot(left_mat, train_y)

preds_theta = np.dot(test_X, theta)

# ===============================
# sklearn regressor
# ===============================
lg_model = LinearRegression()
lg_model.fit(train_X, train_y)
preds_lg = lg_model.predict(test_X).ravel()

sgd_model = SGDRegressor(alpha=1e-4, max_iter=10000, tol=1e-4, eta0=1e-4,
                         learning_rate='invscaling')
sgd_model.fit(train_X, train_y.as_matrix().ravel())
preds_sk = sgd_model.predict(test_X)


# ============================================
# Final comparison (test ds versus all models)
# ============================================

print(mae(test_y, my_preds))
print(mae(test_y, preds_theta))
print(mae(test_y, preds_sk))
print(mae(test_y, preds_lg))

# What is the distribution of the absolute residuals for my model?
#plt.hist(all_preds['truth'] - all_preds['my gd'], bins=30)

# Let's take a closer look at log(residuals)
'''
residuals = np.abs(all_preds['my gd'] - all_preds['truth'])
plt.plot(np.random.choice(np.log(residuals),
                          size=int(1.0 * len(all_preds)), replace=False),
        '.', markersize=1, color='slateblue')
plt.ylim([-2, 16])
'''
