#!/usr/bin/env python

# Inspired by https://www.kaggle.com/igormunizims/pre-processing-for-data-augmentation
# Thanks sir!

import numpy as np
import string
import random

from os import path
from PIL import Image
from skimage.io import imread
from scipy.misc import imresize
from skimage.exposure import adjust_gamma

def crop(image, patch_sz=512):
    '''
    Receives a numpy.ndarray and crops a square patch from
    its centre according to an optionally given patch size.
    '''
    width, height = image.shape

    top    = (height - patch_sz) / 2
    left   = (width  - patch_sz) / 2
    right  = (width  + patch_sz) / 2
    bottom = (height + patch_sz) / 2

    img = Image.fromarray(image)

    return img.crop((left, top, right, bottom))


def generate_random_filepath(basedir='../temp', ext='jpg'):
    '''
    Returns a randomly generated string of 16 characters
    fit to a temporary file name.
    '''
    temp_name = ''.join(random.choice(string.ascii_lowercase +
        string.digits) for _ in range(16))
    
    return path.join(basedir, temp_name) + ext


def jpeg_recompression(image, quality):
    if quality is not int:
        print('quality parametre must be integer. Rounding %f' % (float(quality)))
        quality = np.int(quality)
    
    temp_name = generate_random_filepath()
    img = Image.fromarray(image)
    img.save(temp_name, 'JPEG', quality=quality)

    return imread(temp_name)

def gamma_correction(image, gamma):
    return adjust_gamma(image, gamma)

def resize_bicubic(image, factor):
    '''
    Calls scipy to resize the image using bicubic interpolation
    according to a given factor.
    '''
    return imresize(image, factor, interp='bicubic')

