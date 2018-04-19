#!/usr/bin/env python3

import numpy as np
import glob
import os

from os import path

from scipy import stats

from skimage.io import imread
from skimage.color import rgb2ycbcr
from skimage.restoration import denoise_wavelet

from multiprocessing import Pool

# Pipeline:
# - Extract the 9 RoI from the input image (they are 512x512)
# - For each RoI, use wavelet denoising to extract the noise from R, G, B and Y (luninance from YCbCr)
# - Average each of the 9 RoI noise across all training images for that device
# At the end, you shall have R, G, B and Y noise patterns for 9 regions for all devices,
# i.e. 4 * 9 = 36 noise patterns for each device.
# A feature vector for an image I = {I1, I2, I3, I4, I5, I6, I7, I8, I9} is then composed using all
# calculated correlations between each RoI (4 channels) with the corresponding one from each of the
# known classes.
# So 1 image becomes 9 RoIs. One RoI has 4 colour channels. Each channel gets its noise extracted
# and compared with the other corresponding noise for a candidate camera. THAT'S 36 NOISE PLANES.
# An image has then 36 features, 4 from each of its 9 RoI.

# Proceeding now can lead me to one of the following ways:
# a) I can store the 36 * (number of classes) mean noise patterns. The training is then to input 36
#    noise patterns to a function that calculates the correlations between those 36 for all classes
#    and concatenates them to a vector
# b) I can calculate those correlations but each class is fed to a classifier, so each classifier
#    accepts a 36 correlations vector that I calculated from the input image against each of the
#    36 * (number of classes) noise patterns. Each classifier outputs something like a probability.
#    Sort and pick.
# c) I extract like d features from those 36 noise maps from each image. A classifier is then trained
#    on these features for each labeled image. An input image has it's 36 noise map transformed into
#    the same d features and then given to the classifier, whicih outputs the class.

# Features:
# WHAT ABOUT I START TEH PIPELINE AND DECIDE THESE LATER? OKAY, THANKS FOR LISTENING.

# Patch size (it's always a square AND always a power of 2)
PSZ = 512
# Half PSZ
HPSZ = int(PSZ / 2)

def extract_roi(image):
    '''
    Given an input image, outputs 9 RoI patches with 512x512 pixels as
    done by Costa et al.
    '''
    m, n, c = image.shape

    if c != 3:
        print('Input image had %d colour planes instead of 3.' % (c))

        return None
    
    if not m >= 2 * PSZ and n >= 2 * PSZ:
        print('Input image does not have the minimum dimensions necessary for patch extraction. Skipping.')

        return None

    patches = []

    # UL, UR, LR, LL
    corners = np.array([
        [      0,       0],
        [      0, n - PSZ],
        [m - PSZ, n - PSZ],
        [m - PSZ,       0]
    ])

    for start in corners:
        end = start + PSZ
        patch = image[start[0]:end[0], start[1]:end[1], :]

        patches.append(patch)
    
    # mid point needed by the 5 centre RoI
    r_mid, c_mid = np.floor(np.array([m, n]) / 2)

    r_mid = int(r_mid)
    c_mid = int(c_mid)

    # UL, UR, LR, LL
    centre_4 = np.array([
        [r_mid - PSZ, c_mid - PSZ],
        [r_mid - PSZ,       c_mid],
        [      r_mid, c_mid - PSZ],
        [      r_mid,       c_mid]
    ])

    for start in centre_4:
        end = start + PSZ
        patch = image[start[0]:end[0], start[1]:end[1], :]

        patches.append(patch)

    centre_patch = image[(r_mid - HPSZ):(r_mid + HPSZ),
                         (c_mid - HPSZ):(c_mid + HPSZ), :]

    patches.append(centre_patch)

    return patches


def append_luminance(patch):
    '''
    Input is an image patch as an (R, G, B) vector.
    Output is a vector containing, also, the Y (luminance) channel as
    a vector: (R, G, B, Y). The other planes are not changed.
    '''
    ycbcr_patch = rgb2ycbcr(patch)
    patch = np.dstack((patch, ycbcr_patch[:,:,0]))

    return patch


def get_noise(patch):
    '''
    Input expects an RGBY patch.
    DWT is used to compute a noise mask on each color plane as: noise = input - DWT_denoise(input).
    That's the output.
    '''

    # Possible change of parameters: multichannel to True, wavelet to other type
    denoised_patch = denoise_wavelet(patch, wavelet='db5', convert2ycbcr=False, multichannel=True)

    return patch.astype(np.float64) - denoised_patch


def extract_features(noise_plane):
    '''
    The input is a plane array with a noise pattern.
    Output is a feature vector based on that pattern.
    '''
    desc = stats.describe(noise_plane, axis=None)
    _, _, mean, var, skew, kurt = desc

    feats = np.array([mean, var, skew, kurt])

    return feats


def process_patch_subroutine(p):
    p = append_luminance(p)
    noises = get_noise(p)
    _, _, c = noises.shape

    feats = np.array([extract_features(npl) for npl in np.dsplit(noises, c)])

    return feats.ravel()


def the_algorithm():
    # load m images
    # for each image call extract roi
    # for each roi, call append luminance
    # for each plane of the roi, call get_noise_plane
    # for each noise plane, call extract features
    # concatenate all features from each plane of each roi (36 planes total times number of features)
    # train a classifier with all of those
    # profit

    DATASET_DIR = '../dataset/train'
    #DATASET_DIR = '../problematic-motherfuckers'

    # Each directory inside dataset/ is a class
    classes_dir = glob.glob(path.join(DATASET_DIR, '*'))

    print(classes_dir)

    all_feats = []
    labels = []

    for class_dir in classes_dir:
        image_paths = glob.glob(path.join(class_dir, '*.jpg'))
        class_name = path.basename(class_dir)

        print('Starting class name ' + class_name)
        print('Starting class_dir %s' % (class_dir))

        for i, img_path in enumerate(image_paths):
            print(' Starting image %s\t\t(%d of %d)' % (img_path, i+1, len(image_paths)))
            img = imread(img_path)
            img = img.astype(np.float64)
            patches = extract_roi(img)

            if patches == None:
                # Something went wrong with patch extraction; life goes on
                continue
            
            # Processes each patch individually
            proc_pool = Pool(9)
            results = proc_pool.map(process_patch_subroutine, patches)

            # These block the pool from being used and wait for its threads to terminate
            proc_pool.close()
            proc_pool.join()

            # results contains features/patch, we gotta ravel all 9 image patches
            feats = np.concatenate(results)

            all_feats.append(feats)
            labels.append(class_name)

            # non-parallelised code
            '''
            for p in patches:
                p = append_luminance(p)
                noises = get_noise(p)
                _, _, c = noises.shape

                feats = np.array([extract_features(npl) for npl in np.dsplit(noises, c)]).ravel()

                # does not index labels as integers!
                all_feats.append(feats)
                labels.append(class_name)
            '''
    
    print('done extracting features')

    return [all_feats, labels]

print('done')