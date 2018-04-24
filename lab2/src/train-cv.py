#!/usr/bin/env python3

'''
Objective: train some models and crossvalidate them,
displaying accuracy at the end.
'''

import numpy as np
import itertools
import time
import sys
import warnings

from matplotlib import pyplot as plt

from skimage.io import imread

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier as KNN

from xgboost import XGBClassifier

# To stop sklearn printing stuff on my experiments when MY DATA IS PERFECTLY FINE
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

# Hyperparameters and stuff
scale_zmuv = True
train_stage1 = False
train_stage1_1 = True
train_stage2 = False

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
    X_vanilla = scaler.transform(X_vanilla)
    X_augmented = scaler.transform(X_augmented)

# Let's try a two phase classification process:
#   Phase 1: train 1 classifier per class that tells us if an image is altered or not
#   Phase 2: train a multiclass classifier that classifies unaltered images and another one that classifies altered images

# Stage 1: one binary classifier per class that differentiates unaltered and altered images
if train_stage1:
    class_names = np.unique(y)

    stage1_clfs = dict(zip(class_names, [XGBClassifier(n_jobs=4, max_depth=5, learning_rate=0.1, n_estimators=100, objective='reg:logistic')] * len(class_names)))
    stage1_scores = []

    for class_name, clf in stage1_clfs.items():
        print('CVing stage 1 classifier for class %s...' % (class_name), end=' ')

        # input data: X only for class class_name and labels pristine and tampered
        subX = np.vstack((X_vanilla[y_vanilla == class_name],
                        X_augmented[y_augmented == class_name]))
        suby = np.concatenate((['pristine'] * len(y_vanilla[y_vanilla == class_name]),
                            ['tampered'] * len(y_augmented[y_augmented == class_name])))

        exec_time = time.time()
        score = cross_val_score(clf, subX, y=suby, scoring='accuracy', cv=5, n_jobs=4)
        exec_time = time.time() - exec_time

        stage1_scores.append(score)

        print('Done.\n\tAccuracy = %.2f +- %.2f (CV took %.3f seconds)' % (np.mean(score), np.std(score), exec_time))

    # Stage 1 CV accuracy: ~75%

# Stage 1.1: one binary classifier that differentiates unaltered and altered images
if train_stage1_1:
    clfs = {
        #'LogReg':           LogReg(penalty='l2', max_iter=2000, n_jobs=4, multi_class='ovr', solver='newton-cg'),
        #'SVC':              SVC(kernel='rbf', max_iter=2000, decision_function_shape='ovr'),
        'LSVC':             LinearSVC(penalty='l2', max_iter=2000),
        'SGD LogReg':       SGDClassifier(penalty='l2', learning_rate='optimal', average=True, max_iter=10000, tol=1e-4, loss='log'),
        'SGD LSVC':         SGDClassifier(penalty='l2', learning_rate='optimal', average=True, max_iter=10000, tol=1e-4, loss='hinge'),
        'XGBoost LogReg':   XGBClassifier(n_jobs=4, max_depth=4, learning_rate=0.1, n_estimators=200, objective='reg:logistic'),
        'K-NN':             KNN(n_neighbors=5, n_jobs=4),
        #'XGBoost LinReg':   XGBClassifier(n_jobs=4, max_depth=4, learning_rate=0.1, n_estimators=100, objective='reg:linear'),
        #'XGBoost Softmax':  XGBClassifier(n_jobs=4, max_depth=4, learning_rate=0.1, n_estimators=100, objective='multi:softmax')
    }
    scores1_1 = []

    # Run scoring through CV
    for n, clf in clfs.items():
        print('CVing stage 1.1 classifier %s...' % (n), end=' ')

        # labels are different for this training
        y_s1_1 = np.concatenate((
            ['vanilla'] * len(y_vanilla),
            ['tampered'] * len(y_augmented)
        ))

        exec_time = time.time()
        score = cross_val_score(clf, X,   y=y_s1_1,   scoring='accuracy', cv=5, n_jobs=4)
        exec_time = time.time() - exec_time

        print('Done.\n\tAccuracy = %.2f +- %.2f (CV took %.3f seconds)' % (np.mean(score), np.std(score), exec_time))

        scores1_1.append(score)

# Stage 2: a pair of multiclass classifiers: one for pristine images, one for tampered images
if train_stage2:
    clfs = {
        'LogReg':           [LogReg(penalty='l2', max_iter=2000, n_jobs=4, multi_class='ovr', solver='newton-cg')] * 2,
        'SVC':              [SVC(kernel='rbf', max_iter=2000, decision_function_shape='ovr')] * 2,
        'LSVC':             [LinearSVC(penalty='l2', multi_class='ovr', max_iter=2000)] * 2,
        'SGD LogReg':       [SGDClassifier(penalty='l2', learning_rate='optimal', average=True, max_iter=10000, tol=1e-4, loss='log')] * 2,
        'SGD LSVC':         [SGDClassifier(penalty='l2', learning_rate='optimal', average=True, max_iter=10000, tol=1e-4, loss='hinge')] * 2,
        'XGBoost LogReg':   [XGBClassifier(n_jobs=4, max_depth=4, learning_rate=0.1, n_estimators=200, objective='reg:logistic')] * 2,
        #'XGBoost LinReg':   XGBClassifier(n_jobs=4, max_depth=4, learning_rate=0.1, n_estimators=100, objective='reg:linear'),
        #'XGBoost Softmax':  XGBClassifier(n_jobs=4, max_depth=4, learning_rate=0.1, n_estimators=100, objective='multi:softmax')
    }
    scores1 = []
    scores2 = []

    # Run scoring through CV
    for n, [c1, c2] in clfs.items():
        print('CVing stage 2 classifiers %s...' % (n), end=' ')

        exec_time = time.time()
        score1 = cross_val_score(c1, X_vanilla,   y=y_vanilla,   scoring='accuracy', cv=5, n_jobs=4)
        score2 = cross_val_score(c2, X_augmented, y=y_augmented, scoring='accuracy', cv=5, n_jobs=4)
        exec_time = time.time() - exec_time

        print('Done.\n\tAccuracies = (%.2f, %.2f) +- (%.2f, %.2f) (CV took %.3f seconds)' % (
            np.mean(score1), np.mean(score2), np.std(score1), np.std(score2), exec_time))

        scores1.append(score1)
        scores2.append(score2)

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
