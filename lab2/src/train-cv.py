#!/usr/bin/env python3

'''
Objective: train some models and crossvalidate them,
displaying accuracy at the end.
'''

import numpy as np

from matplotlib import pyplot as plt

from skimage.io import imread

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC, LinearSVC

# Load features
X = np.load('../data/noise-features2-dwt-norm/512x512patch_rgby_40corrs.npy')
y = np.load('../data/noise-features2-dwt-norm/512x512patch_rgby_40corrs_labels.npy')

# Transform X to a matrix because I forgot to do it
X = np.matrix(X)

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
