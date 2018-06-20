import os
import sys
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from tqdm import tqdm

from matplotlib import pyplot as plt

from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing import image
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Dense, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.callbacks import Callback, LearningRateScheduler

from sklearn.metrics import accuracy_score, f1_score

from tensorflow import train

from utils import ask_user

####################################################
#                    PARAMETERS
####################################################

# These are some defaults that might change
# according to chosen architecture

train_dir = 'data/train/'
test_dir = 'data/test/'
val_dir = 'data/val/'

# image_size MUST AGREE with receptive_field
params = {
    'image_size':         (299, 299),
    'receptive_field':    (299, 299, 3),
    'batch_size':         32,
    'n_classes':          83,
    'n_epochs':           10,
    'learning_rate_sgd1': 1e-4,
    'learning_rate_sgd2': 0.045,
    'adam': {
        'learning_rate': 0.001,
        'beta1':   0.9,
        'beta2':   0.999,
        'epsilon': None,
        'decay':   0.0,
    },
    'optimizer':          'sgd',
    'lr_scheduling':      None
}

########################################################################
#                          IMAGE AUGMENTATION
########################################################################

# Because net was trained for imagenet, I should normalise the input using:
# mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
# However, results came even without it, so I am sticking to no normalisation
# until I find it necessary.

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    zoom_range=0.2
)

validation_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=params['image_size'],
    batch_size=params['batch_size'],
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=params['image_size'],
    batch_size=params['batch_size'],
    class_mode='categorical'
)

if ask_user('Data generators created. Continue to Network Initialisation?') == False:
    sys.exit(0)

########################################################################
#                        NETWORK INITIALISATION
########################################################################
net = InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=params['receptive_field']
)

# Disable training in all but the last 4 layers
for layer in net.layers:
    layer.trainable = False

#for layer in net.layers[-17:]:
#    layer.trainable = True

# And check it
#for layer in net.layers:
#    print(layer, layer.trainable)

model = Sequential()

model.add(net)
model.add(GlobalAveragePooling2D(name='global_avg_pool'))
model.add(Dense(params['n_classes'], activation='softmax', name='classifier'))

model.summary()

#optimizer = optimizers.SGD(lr=params['learning_rate_sgd1'], momentum=0.9, decay=1e-6)
#optimizer = optimizers.SGD(lr=0.045, momentum=0.9, decay=0.45) # Google's (0.94 decay each 2 epochs though)
optimizer = optimizers.Adam(
    lr=params['adam']['learning_rate'],
    beta_1=params['adam']['beta1'],
    beta_2=params['adam']['beta2'],
    epsilon=params['adam']['epsilon'],
    decay=params['adam']['decay'],
    amsgrad=False
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['categorical_accuracy']
)

if ask_user('Model is ready. Continue to training?') == False:
    sys.exit(0)

################################################################################
#                                   TRAINING
################################################################################
# This will hold the history objects
history_df = pd.DataFrame()

# The training, per se
train_history = model.fit_generator(
    train_generator,
    epochs=params['n_epochs'],
    validation_data=validation_generator,
    shuffle=True
)

append_this = pd.DataFrame.from_dict(train_history.history)
history_df = history_df.append(append_this)

# This remakes the index so it doesn't contain repeated ordinals
history_df = history_df.reset_index()

if ask_user('Training is complete. Continue to plotting?', default='no') == False:
    sys.exit(0)

########################################################
#                       PLOTTING
########################################################
acc = history_df['categorical_accuracy']
val_acc = history_df['val_categorical_accuracy']
loss = history_df['loss']
val_loss = history_df['val_loss']

epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()

model.save('./inceptionv_v3_spoton.model')
np.save('./inception_v3_spoton_history.npy', train_history.history)

print('done')
