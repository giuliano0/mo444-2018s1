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

from xgboost import XGBClassifier

# To stop sklearn printing stuff on my experiments when MY DATA IS PERFECTLY FINE
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

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
    X_vanilla = scaler.transform(X_vanilla)
    X_augmented = scaler.transform(X_augmented)

# Let's try a two phase classification process:
#   Phase 1: train 1 classifier per class that tells us if an image is altered or not
#   Phase 2: train a multiclass classifier that classifies unaltered images and another one that classifies altered images

# Stage 1: one binary classifier per class that differentiates unaltered and altered images
class_names = np.unique(y)

stage1_clfs = dict(zip(class_names, [XGBClassifier(n_jobs=4, max_depth=5, learning_rate=0.1, n_estimators=100, objective='reg:logistic')] * len(class_names)))

for class_name, clf in stage1_clfs.items():
    print('CVing stage 1 classifier for class %s...' % (class_name), end=' ')

    # input data: X only for class class_name and labels pristine and tampered
    subX = np.vstack((X_vanilla[y_vanilla == class_name],
                    X_augmented[y_augmented == class_name]))
    suby = np.concatenate((['pristine'] * len(y_vanilla[y_vanilla == class_name]),
                        ['tampered'] * len(y_augmented[y_augmented == class_name])))

    exec_time = time.time()
    clf.train(subX, suby)
    exec_time = time.time() - exec_time

    print('Done. (Training took %.3f seconds)' % (exec_time))

# Stage 2: a pair of multiclass classifiers: one for pristine images, one for tampered images
clfs = {
    'LogReg':           [LogReg(penalty='l2', max_iter=2000, n_jobs=4, multi_class='ovr', solver='newton-cg')] * 2
}

for n, [c1, c2] in clfs.items():
    print('CVing stage 2 classifiers %s...' % (n), end=' ')

    exec_time = time.time()
    c1.train(X_vanilla,   y=y_vanilla)
    c2.train(X_augmented, y_augmented)
    exec_time = time.time() - exec_time

    print('Done. (Training took %.3f seconds)' % (exec_time))

# Load and normalise test data
testX = np.load('../data/lbp-RBW/512x512patch_rbw_lbp_test.npy')
test_fnames = np.load('../data/lbp-RBW/512x512patch_rbw_lbp_test_labels.npy')

testX = scaler.transform(testX)

# Predicts if images were altered or not
image_state_preds = None # TODO

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
