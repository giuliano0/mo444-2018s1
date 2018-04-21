#!/usr/bin/env python3

import numpy as np
import glob
import os
import sys

from os import path

from scipy import stats
from scipy.signal import correlate2d

from skimage.io import imread
from skimage.color import rgb2ycbcr
from skimage.restoration import denoise_wavelet
from skimage.filters import median
from skimage.feature import greycomatrix, greycoprops

from multiprocessing import Pool

# Patch size (it's always a square AND always a power of 2)
PSZ = 512
# Half PSZ
HPSZ = int(PSZ / 2)

def extract_patch(image):
    '''
    Given an input image, outputs 9 RoI patches with 512x512 pixels as
    done by Costa et al.
    '''
    m, n, c = image.shape

    if c != 3:
        print('Input image had %d colour planes instead of 3.' % (c))

        return None
    
    if not m >= PSZ and n >= PSZ:
        print('Input image does not have the minimum dimensions necessary for patch extraction. Skipping.')

        return None

    # Test images are all (512, 512, 3) I guess,
    # so just give back the image itself and that's it
    if m == PSZ and n == PSZ:
        return image

    # mid point needed by the 5 centre RoI
    r_mid, c_mid = np.floor(np.array([m, n]) / 2)

    r_mid = int(r_mid)
    c_mid = int(c_mid)

    centre_patch = image[(r_mid - HPSZ):(r_mid + HPSZ),
                         (c_mid - HPSZ):(c_mid + HPSZ), :]

    return centre_patch


def append_luminance(image):
    '''
    Input is an image patch as an (R, G, B) vector.
    Output is a vector containing, also, the Y (luminance) channel as
    a vector: (R, G, B, Y). The other planes are not changed.
    '''
    ycbcr_image = rgb2ycbcr(image)
    image = np.dstack((image, ycbcr_image[:,:,0]))

    return image


def get_noise(patch):
    '''
    Input expects an RGBY patch.
    DWT is used to compute a noise mask on each color plane as: noise = input - DWT_denoise(input).
    That's the output.
    '''
    _, _, channels = patch.shape
    denoised_patch = np.zeros(patch.shape)

    for c in range(channels):
        denoised_patch[:,:,c] = median(patch[:,:,c].astype(np.uint8), selem=None)

    # I'm gonna normalise this in the gaussian case
    denoised_patch = (patch - denoised_patch)

    #denoised_patch -= np.min(denoised_patch, axis=(0, 1))   # min across planes
    #denoised_patch /= np.max(denoised_patch, axis=(0, 1))   # max across planes

    if np.min(denoised_patch) < 0 or np.max(denoised_patch) > 255:
        print('denoised_patch outside 8-bit image range (min, max) = (%f, %f), Renormalising.' %
            (np.min(denoised_patch), np.max(denoised_patch)))
        
        denoised_patch -= np.min(denoised_patch.astype(np.float64), axis=(0, 1))   # min across planes
        denoised_patch /= np.max(denoised_patch, axis=(0, 1))   # max across planes
        denoised_patch = 255 * denoised_patch

    return denoised_patch.astype(int)


def process_image_subroutine(img_path):
    print(' Starting image %s' % (img_path), end='\r')

    img = imread(img_path, plugin='matplotlib')
    patch = extract_patch(img)
    patch = append_luminance(patch)
    noises = get_noise(patch)

    feats = []
    _, _, channels = noises.shape
    angles = [a * np.pi / 4 for a in range(4)]

    for c in range(channels):
        prop_list = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        glcm = greycomatrix(noises[:,:,c], [1, 2], angles, levels=256) # (256, 256, 1, 4) array

        for prop in prop_list:
            props = greycoprops(glcm, prop=prop)
            feats.append(props.ravel()) # raveling makes it work for multi-distance props

    # 4 channels * 1 glcm/channel * (4 * 2) distances-angles/glcm * 5 properties/dist-angle
    # 160 features by patch
    return np.ravel(feats)


def the_algorithm(dataset_path=None, is_test=False):
    if dataset_path is None:
        #DATASET_DIR = '../dataset/train'
        DATASET_DIR = '../problematic-motherfuckers'
    else:
        DATASET_DIR = dataset_path

    # Let's extract for each class
    # Then save the SPN reference (1 guy per class) and each image SPN (m images per class)
    # Then we load 1 SPN reference and load each class SPNs and calculate correlations, save results
    # Repeat for all SPN references
    # Open each result file, concatenate
    # Profit

    # Each directory inside dataset/train/ is a class
    classes_dir = glob.glob(path.join(DATASET_DIR, '*'))

    print('STAGE 1: FEATURE EXTRACTION')

    all_feats = []
    all_labels = []

    for cidx, class_dir in enumerate(classes_dir):
        if not is_test:
            file_pattern = '*.[Jj][Pp][Gg]'
        else:
            file_pattern = '*.[Tt][Ii][Ff]'

        class_name = path.basename(class_dir)
        image_paths = glob.glob(path.join(class_dir, file_pattern))
        proc_pool = Pool(4)

        print('Starting class \"%s\" (%d of %d)' % (class_name, cidx + 1, len(classes_dir)))
        print('class_dir %s' % (class_dir))

        try:
            noise_patterns = proc_pool.map(process_image_subroutine, image_paths)
            labels = [class_name] * len(noise_patterns)
        except:
            raise
            sys.exit(1)

        all_feats.append(noise_patterns)
        all_labels += labels

        # Saving in case we need to concatenate them afterwards (out of RAM issue)
        np.save('../data/median_%s' % (class_name), noise_patterns)

        if is_test:
            np.save('../data/%s_labels' % (class_name), image_paths)
        else:
            np.save('../data/%s_labels' % (class_name), labels)

    print('done extracting features')

    return [np.matrix(all_feats), all_labels]

print('done')
