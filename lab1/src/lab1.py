import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

train_X = pd.read_csv('../data/train.csv')
train_y = train_X['shares']

# Feature selection
train_X.drop('shares', axis=1, inplace=True)
train_X.drop('url', axis=1, inplace=True)

# I think we don't need feats to be float64
train_X = train_X.astype(np.float32)
train_y = train_y.astype(np.float32)

# some features are scaled to not-so-reasonable ranges
train_X['timedelta'] = train_X['timedelta'].apply(lambda x: x / 10)
train_X['n_tokens_content'] = train_X['n_tokens_content'].apply(lambda x: x / 10)
train_X['kw_max_min'] = train_X['kw_max_min'].apply(lambda x: x / 100)
train_X['kw_avg_min'] = train_X['kw_avg_min'].apply(lambda x: x / 10)
train_X['kw_min_max'] = train_X['kw_min_max'].apply(lambda x: x / 1000)
train_X['kw_max_max'] = train_X['kw_max_max'].apply(lambda x: x / 10000)
train_X['kw_avg_max'] = train_X['kw_avg_max'].apply(lambda x: x / 10000)
train_X['kw_min_avg'] = train_X['kw_min_avg'].apply(lambda x: x / 100)
train_X['kw_max_avg'] = train_X['kw_max_avg'].apply(lambda x: x / 100)
train_X['kw_avg_avg'] = train_X['kw_avg_avg'].apply(lambda x: x / 100)
train_X['self_reference_min_shares'] = train_X['self_reference_min_shares'].apply(lambda x: x / 100)
train_X['self_reference_max_shares'] = train_X['self_reference_max_shares'].apply(lambda x: x / 1000)
train_X['self_reference_avg_sharess'] = train_X['self_reference_avg_sharess'].apply(lambda x: x / 100)

m, n = train_X.shape

# tf workd with batches
batch_size = 50
# we also need learning rate and iterations and step (for logging only)
learning_rate = 0.00001
iterations = 1
step = 20

# init placeholders
x = tf.placeholder(dtype=tf.float32, name='x_i')
#x = tf.placeholder(shape=[None, n], dtype=tf.float32, name='x_i')
y = tf.placeholder(dtype=tf.float32, name='y_i')

# init regression variables
w = tf.Variable(tf.random_normal(shape=[n,1]), name='weights')
b = tf.Variable(tf.random_normal(shape=[1,1]), name='bias')

# hypothesis: y = <w, x> + b
hypothesis = tf.add(tf.matmul(x, w), b)

# loss: MSE (could be implemented w/ sum and divided by 2*m but who cares?)
loss = tf.reduce_mean(tf.square(y - hypothesis))

# now the optimizer
gd = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# tf also needs to init globals, w/e this is
init = tf.global_variables_initializer()

# our track variables for loss
training_loss = []
test_loss = []

# now we create a session and start coding
with tf.Session() as sess:
    sess.run(init)
    
    for it in range(iterations):
        # this fits for the entire data
        for (x_i, y_i) in zip(train_X.as_matrix(), train_y.as_matrix()):
            x_i = x_i[:, np.newaxis].T
            
            sess.run(gd, feed_dict={x: x_i, y: y_i})
        
        # calculates and stores loss for this it
        loss_it = sess.run(loss, feed_dict={x: train_X.as_matrix(), y: train_y.as_matrix()})
        
        training_loss.append(loss_it)
        
        # display training loss logs each _step_ iterations
        if (it % step) == 0:
            print("Iteration %04d, loss %.3f" % (it, loss_it))

    print("done")
    
    # now we calculate test_loss




# train as is
# normalize each feature and train again
