import tensorflow as tf

from keras import Sequential
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Softmax
from keras.optimizers import SGD
from keras import backend as K

# TODO: implement checkpoints using callbacks (I think it's in the keras docs)
#from keras.callbacks import g

'''
Images are broken in 64x64 patches
Patches with saturated pixels are ignored
Patch pixels are subtracted by the pixel average over the training set
Patches are pixel normalised by 0.0125

General sequential model as described in the paper:
- conv w/ 32 filters 4x4x3 stride 1
- max pool kernel 2 stride 2
- conv w/ 48 filters 5x5x32 stride 1
- max pool kernel 2 stride 2
- conv w/ 64 filters 5x5x48 stride 1 
- max pool kernel 2 stride 2
- conv w/ 128 filters 5x5x64 stride 1: outputs 128-vector
- inner product 128 neurons
- relu output 128-vector
- final 128xN inner product, N number of classes
- softmax

trained by SGD with 128 patches batch size

So, to augment this data I can take batches of say, 16 images and
extract 8 patches from each to ramp it up to 128 like they did in the paper

This way I can use a vanilla ImageGenerator and augment it on the fly
'''
def buildModel():
    # these are patch dimensions
    image_width = 64
    image_height = 64

    if K.image_data_format() == 'channels_first':
        input_shape = (3, image_width, image_height)
    else:
        input_shape = (image_width, image_height, 3)

    model = Sequential()

    model.add(Convolution2D(32, (4, 4), strides=(1, 1), padding='same', input_shape=input_shape, name='conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1'))

    model.add(Convolution2D(48, (5, 5), strides=(1, 1), padding='same', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2'))

    model.add(Convolution2D(64, (5, 5), strides=(1, 1), padding='same', name='conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3'))

    model.add(Convolution2D(128, (5, 5), strides=(1, 1), padding='same', name='conv4'))
    
    model.add(Dense(128, name='ip1'))
    model.add(Activation('relu', name='relu1'))

    #model.add(Flatten())
    model.add(Dense(18, name='ip2'))
    model.add(Softmax(name='softmax'))

    return model

from keras.optimizers import SGD

def prepareModel(model):
    opt = SGD(lr=0.003, momentum=0.9, decay=1e-6)
    
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

    return model

import numpy as np
import tensorflow as tf

# SomethingI guell I'll hae to implement myself:
class PatchGenerator(object):
    def __init__(self, X, y, batch_size=32):
        self.batch_size = batch_size
        self.X = X # list of input images
        self.y = y # list of classes for X images
        self.index = 0
    
    def __iter__(self):
        return self
    
    def next(self):
        image = self.X[self.index]
        target = self.y[self.index]
        self.index += 1

        if self.index > len(self.X):
            self.index = 0

        batch = []

        for i in range(self.batch_size):
            patch = tf.extract_image_patches(self.X)
            
            batch.append((patch, target))

        return np.array(batch)

from keras.preprocessing.image import ImageDataGenerator

def trainModel(model):
    print('Building data generators...')

    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        'data/train', batch_size=32
    )
    
    validation_generator = test_datagen.flow_from_directory(
        'data/validaton', batch_size=32
    )

    # example with my generator:
    gen = PatchGenerator(X, y)
    model.fit_generator(gen, samples_per_epoch=10000) # samples_per_epoch=len(X)


# Example model I dropped here to test net output
from keras.layers import Dropout

def createModel():
    nClasses = 10

    # these are patch dimensions
    image_width = 32
    image_height = 32

    if K.image_data_format() == 'channels_first':
        input_shape = (3, image_width, image_height)
    else:
        input_shape = (image_width, image_height, 3)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
     
    return model
