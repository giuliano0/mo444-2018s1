#!/usr/bin/env python3

'''
Objective: train some models and crossvalidate them,
displaying accuracy at the end.
'''

import numpy as np
import itertools
import time

from matplotlib import pyplot as plt

from skimage.io import imread

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, LinearSVC

from xgboost import XGBClassifier

# Hyperparameters and stuff
scale_zmuv = True

# Load features
X_vanilla = np.load('../data/lbp-RBW/512x512patch_rbw_lbp.npy')
y_vanilla = np.load('../data/lbp-RBW/512x512patch_rbw_lbp_labels.npy')
# Augmented set
X_augmented = np.load('../data/lbp-RBW-augmented-set/512x512patch_rbw_lbp_augmented.npy')
y_augmented = np.load('../data/lbp-RBW-augmented-set/512x512patch_rbw_lbp_augmented_labels.npy')

# Merge all data
X = np.vstack((X_vanilla, X_augmented))
y = np.concatenate((y_vanilla, y_augmented))

print('Input data has shape %s' % (str(X.shape)))

if scale_zmuv:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

# Declare classifiers
clfs = {
    'LogReg':           LogReg(penalty='l2', max_iter=2000, n_jobs=4, multi_class='ovr', solver='newton-cg'),
    'SVC':              SVC(kernel='rbf', max_iter=2000, decision_function_shape='ovr'),
    'LSVC':             LinearSVC(penalty='l2', multi_class='ovr', max_iter=2000),
    'SGD LogReg':       SGDClassifier(penalty='l2', learning_rate='optimal', average=True, max_iter=10000, tol=1e-4, loss='log'),
    'SGD LSVC':         SGDClassifier(penalty='l2', learning_rate='optimal', average=True, max_iter=10000, tol=1e-4, loss='hinge'),
    'XGBoost LogReg':   XGBClassifier(n_jobs=4, max_depth=4, learning_rate=0.1, n_estimators=200, objective='reg:logistic'),
    #'XGBoost LinReg':   XGBClassifier(n_jobs=4, max_depth=4, learning_rate=0.1, n_estimators=100, objective='reg:linear'),
    #'XGBoost Softmax':  XGBClassifier(n_jobs=4, max_depth=4, learning_rate=0.1, n_estimators=100, objective='multi:softmax')
}
scores = []

# Classifier    acc   std     Commentary
# -----------   ----  ----  -----------
# LogReg        0.67  0.12  takes a lifetime to converge using SAG, 1000 its is not enough
# SVC           0.56  0.13  crappy
# LSVC          0.66  0.11  hm...
# SGD LogReg    0.64  0.12  kinda promising
# SGD LSVC      0.63  0.12  worse than regular LSVC
# XGB LogReg    0.71  0.11  takes a lifetime to converge with these parameters and reported an error during CV
# XGB LinReg    0.71  0.11  takes a lifetime to converge with these parameters and reported an error during CV
# XGB Softmax   0.71  0.11  takes a lifetime to converge with these parameters and reported an error during CV

# All XGB were performing the same, so I disabled 2 of them
# LogReg still takes a lifetime to converge

# Run scoring through CV
for n, c in clfs.items():
    print('CVing %s...' % (n), end=' ')

    exec_time = time.time()
    score = cross_val_score(c, X, y=y, scoring='accuracy', cv=5, n_jobs=4)
    exec_time = time.time() - exec_time

    print('Done. Accuracy = %.2f +- %.2f (CV took %.3f seconds)' % (np.mean(score), np.std(score), exec_time))

    scores.append(score)

# Print results
print('Finished. Variable scores holds the results.')

# Mischief managed

# Test gotta be loaded and saved for submission
#import pandas as pd
#df = pd.DataFrame(list(zip(testY, preds)), columns=('fname', 'camera'))
#df.to_csv('../submissions/noise-features2-gauss.1.csv', index=False)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
