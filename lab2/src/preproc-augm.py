#!/usr/bin/env python

# Inspired by https://www.kaggle.com/igormunizims/pre-processing-for-data-augmentation
# Thanks sir!

import numpy as np
import string
import random
import glob

from os import path, makedirs, remove
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

    if width < patch_sz or height < patch_sz:
        print('Image is too small, skipping.')
        return None
    
    if width == patch_sz and height == patch_sz:
        print('Image is exactly patch size. No crop.')
        return image

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


def jpeg_recompression(image, quality, save_final=None):
    # Make sure quality is int
    quality = np.int(quality)
    
    # If save_final is None, save at a random place and return the image,
    # else save at save_final path and return None
    if save_final == None:
        save_path = generate_random_filepath()
    else:
        save_path = save_final

    img = Image.fromarray(image)
    img.save(save_path, 'JPEG', quality=quality)

    img = imread(save_path)

    if save_final == None:
        remove(save_path)
        return img
    else:
        return None

def gamma_correction(image, gamma):
    return adjust_gamma(image, gamma)

def resize_bicubic(image, factor):
    '''
    Calls scipy to resize the image using bicubic interpolation
    according to a given factor.
    '''
    return imresize(image, factor, interp='bicubic')


def dir_check_create(dir_path):
    if not path.exists(dir_path):
        makedirs(dir_path)
    
    return

# ==================================================
#             DATASET AUGMENTATION
# Takes images train set and augments them according
# to competition parametres.
# 
# Inputs from:   dataset/train
# Outputs to:    dataset/augmented-train
# ==================================================

print('I solemnly swear that I am up to no good.')

rng = np.random

#input_dir  = '../problematic-motherfuckers'
input_dir = '../dataset/train'
output_dir = '../dataset/augmented-train'
temp_dir   = '../temp'

# Directory does not exist, create it
dir_check_create(output_dir)
dir_check_create(temp_dir)

# PARAMETERS
augmentation_ops = ['jpeg recompression', 'bicubic resizing', 'gamma correction']
jpeg_qualities   = [70, 90]
resize_factors   = [0.5, 0.8, 1.5, 2.0]
gamma_factors    = [0.8, 1.2]
aug_prob         = 0.4      # Probability of augmentation (one roll per augmentation)
aug_fn_table = dict(zip(augmentation_ops, [jpeg_recompression, resize_bicubic, gamma_correction]))

# Outter operation: a class
class_dirs = glob.glob(path.join(input_dir, '*'))

for cidx, class_dir in enumerate(class_dirs):
    class_name = path.basename(class_dir)

    print('Starting class %s (%d of %d)...' % (class_name, cidx + 1, len(class_dirs)))

    image_paths = glob.glob(path.join(class_dir, '*.[Jj][Pp][Gg]')) # MFing case sensitiveness

    # Inner operation: an image
    for iidx, image_path in enumerate(image_paths):
        print('  Starting image %s (%d of %d)...' % (image_path, iidx + 1, len(image_paths)))

        dir_check_create(path.join(output_dir, class_name))

        base_filename = '.'.join(path.basename(image_path).split('.')[:-1])
        augmented_base_path = path.join(output_dir, class_name, base_filename)
        img = imread(image_path, plugin='matplotlib') # matplotlib plugin because Sony sucks at saving JPEGs

        # No fancy loops, just one after the other

        # JPEG compression
        if rng.rand() <= aug_prob:
            qual = int(random.choice(jpeg_qualities))
            final_filename = augmented_base_path + ('_jpeg%d.jpg' % (qual))

            print('    Image rolled for JPEG recompression with quality %d' % (qual))

            jpeg_recompression(img, qual, save_final=final_filename)
        
        # Resizing
        if rng.rand() <= aug_prob:
            factor = random.choice(resize_factors)
            final_filename = augmented_base_path + ('_resize%.1f.jpg' % (factor))

            print('    Image rolled for bicubic resizing with factor %.1f' % (factor))

            img2 = Image.fromarray(resize_bicubic(img, factor))

            img2.save(final_filename, 'JPEG', quality=95)

        # Gamma correction
        if rng.rand() <= aug_prob:
            factor = random.choice(gamma_factors)
            final_filename = augmented_base_path + ('_gamma%.1f.jpg' % (factor))

            print('    Image rolled for gamma correction with factor %.1f' % (factor))

            img2 = Image.fromarray(gamma_correction(img, factor))

            img2.save(final_filename, 'JPEG', quality=95)

print('Mischief managed.')
