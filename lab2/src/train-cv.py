#!/usr/bin/env python3

'''
Objective: train some models and crossvalidate them,
displaying accuracy at the end.
'''

import numpy as np
import itertools

from matplotlib import pyplot as plt

from skimage.io import imread

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC, LinearSVC

# Hyperparameters and stuff
scale_zmuv = True

# Load features
X = np.load('../data/lbp-RBW/512x512patch_rbw_lbp.npy')
y = np.load('../data/lbp-RBW/512x512patch_rbw_lbp_labels.npy')

# Transform X to a matrix because I forgot to do it
X = np.matrix(X)

if scale_zmuv:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

# Declare classifiers
lrc = LogReg(penalty='l2', max_iter=1000, n_jobs=4, multi_class='ovr', solver='liblinear')
svcc = SVC(kernel='rbf', max_iter=-1, decision_function_shape='ovr')
lsvcc = LinearSVC(penalty='l2', multi_class='ovr', max_iter=1000)

clfs = [lrc, svcc, lsvcc]
scores = []

# Run scoring through CV
for i, c in enumerate(clfs):
    print('Starting classifier %d' % (i))

    score = cross_val_score(c, X, y=y, scoring='accuracy', cv=5, n_jobs=4)

    scores.append(score)

# Print results
print('Finished. Variable scores holds the results. Printing means and stds:')

for i, s in enumerate(scores):
    print('Classifier %d scored %.2f +- %.2f' % (i, np.mean(s), np.std(s)))

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
