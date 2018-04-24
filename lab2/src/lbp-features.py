#!/usr/bin/env python3

import numpy as np
import glob
import os
import pywt
import sys

from os import path

from scipy import stats

from skimage.io import imread
from skimage.color import rgb2ycbcr
from skimage.restoration import denoise_wavelet     # unused
from skimage.filters import gaussian                # unused

from skimage.feature import local_binary_pattern

from multiprocessing import Pool, Process, Pipe

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

    # mid point needed by the 5 centre RoI
    r_mid, c_mid = np.floor(np.array([m, n]) / 2)

    r_mid = int(r_mid)
    c_mid = int(c_mid)

    centre_patch = image[(r_mid - HPSZ):(r_mid + HPSZ),
                         (c_mid - HPSZ):(c_mid + HPSZ), :]

    return centre_patch


# Do not use (yet)
def append_luminance(image):
    '''
    Input is an image patch as an (R, G, B) vector.
    Output is a vector containing, also, the Y (luminance) channel as
    a vector: (R, G, B, Y). The other planes are not changed.
    '''
    ycbcr_image = rgb2ycbcr(image)
    image = np.dstack((image, ycbcr_image[:,:,0]))

    return image


# Do not use
def get_noise(patch):
    '''
    Input expects an RGBY patch.
    DWT is used to compute a noise mask on each color plane as: noise = input - DWT_denoise(input).
    That's the output.
    '''

    # Possible change of parameters: multichannel to True, wavelet to other type
    #denoised_patch = denoise_wavelet(patch, wavelet='db5', convert2ycbcr=False, multichannel=True)
    denoised_patch = gaussian(patch.astype(np.float64), sigma=0.8, multichannel=True)

    # I'm gonna normalise this in the gaussian case
    denoised_patch = (patch.astype(np.float64) - denoised_patch)

    denoised_patch -= np.min(denoised_patch, axis=(0, 1))   # min across planes
    denoised_patch /= np.max(denoised_patch, axis=(0,1))    # max across planes

    return denoised_patch


# parallel impl
#def calc_prediction_coding(channel0, conn):
def calc_prediction_coding(channel):
    channel_pc = np.zeros(channel.shape, dtype=np.float64)
    # Pad channel to calculate predictive coding
    #channel = np.copy(channel0) # parallel impl
    channel = np.pad(channel, 1, 'constant', constant_values=0)[:-1, :-1]

    # Calculate the residual from the predictive coding directly
    # In a double for. Shame.
    for row in range(1, channel.shape[0] - 1):
        for column in range(1, channel.shape[1] - 1):
            a = channel[row - 1][column]
            b = channel[row][column - 1]
            c = channel[row - 1][column - 1]

            if c <= np.min([a, b]):
                channel_pc[row-1][column-1] = np.max([a, b])
            elif c > np.max([a, b]):
                channel_pc[row-1][column-1] = np.min([a, b])
            else:
                channel_pc[row-1][column-1] = float(a) + float(b) - float(c)
            
            channel_pc[row-1][column-1] = channel[row-1][column-1] - channel_pc[row-1][column-1]
    
    # parallel impl
    #conn.send(channel_pc)

    return channel_pc


# Either R or B comes as channel.
# We take the DWT and from it the HH band
# We calculate the predictive image and then the predictive error PE
# We calculate LBP on channel, HH and PE
# We concatenate them and return a vector
def process_channel(channel):
    # Spawn a process to calculate the predictive coding of channel
    #parent_conn, child_conn = Pipe()
    #pred_coding_proc = Process(name='Predictive coding process',
    #    target=calc_prediction_coding, args=(channel, child_conn))
    
    # Meanwhile, we calculate LBPs for channel and its DWT HH band
    _, (_, _, HH) = pywt.dwt2(channel, 'haar')

    wavelet_lbp, _ = np.histogram(local_binary_pattern(HH, 8, 1, method='uniform'), bins=59)
    spatial_lbp, _ = np.histogram(local_binary_pattern(channel, 8, 1, method='uniform'), bins=59)

    pred_coding = calc_prediction_coding(channel)

    # Now we wait for the predictive coding to finish,
    # take its return value and calculate its LBP
    #pred_coding_proc.join()

    #pred_coding = parent_conn.recv()
    predc_lbp, _ = np.histogram(local_binary_pattern(pred_coding, 8, 1, method='uniform'), bins=59)

    feats = [wavelet_lbp, spatial_lbp, predc_lbp]

    return np.concatenate(feats) # 3 * 59


# We receive an image path and load the image
# We take the centre patch from it and delete the G channel
# Pass it to process_channel and receive back a feature vector for R and one for B
# We concatenate them in a 354-dimensional feature vector
def process_image_subroutine(img_path):
    print(' Starting image %s' % (img_path), end='\r')

    img = imread(img_path, plugin='matplotlib')
    patch = extract_patch(img)
    
    # We now remove the green channel and let patch in the form of a list
    # so the thread pool can operate on ints channels individually
    patch = np.dsplit(patch, patch.shape[2])
    result = []

    del patch[1] # delete green channel

    # A reshape is necessary, the last dimension still exists
    for k in range(len(patch)):
        patch[k] = np.reshape(patch[k], patch[k].shape[:-1])
        
        result.append(process_channel(patch[k]))

    return np.concatenate(result) # 2 * 3 * 59


def extract_features_from_class(class_dir, is_test=False):
    '''
    Returns a list with as many elements as there are images in class_dir.
    Each element is a (512, 512, 4) numpy.ndarray of float64 containing
    each image's estimated SPN.
    '''
    if not is_test:
        file_pattern = '*.[Jj][Pp][Gg]'
    else:
        file_pattern = '*.[Tt][Ii][Ff]'
    
    image_paths = glob.glob(path.join(class_dir, file_pattern)) # MFing case sensitiveness

    proc_pool = Pool(4)

    # Let's try this in parallel
    feats = proc_pool.map(process_image_subroutine, image_paths)

    # Close pool and wait for task completion
    proc_pool.close()
    proc_pool.join()

    print('') # prints nothing but adds a \n to the end of the line
    
    if not is_test:
        return feats, None
    else:
        return feats, image_paths


def the_algorithm(dataset_path=None, is_test=False):
    if dataset_path is None:
        DATASET_DIR = '../dataset/train'
        #DATASET_DIR = '../problematic-motherfuckers'
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

    print(classes_dir)

    print('STAGE 1: SPN EXTRACTION (CLASS REFERENCE AND ALL IMAGES)')

    for cidx, class_dir in enumerate(classes_dir):
        class_name = path.basename(class_dir)

        print('Starting class \"%s\" (%d of %d)' % (class_name, cidx + 1, len(classes_dir)))
        print('class_dir %s' % (class_dir))

        features, labels_or_filenames = extract_features_from_class(class_dir, is_test)

        # No need to save labels. I have them by the class_name variable
        np.save('../data/lbpfeats_%s' % (class_name), features)

        if is_test:
            np.save('../data/lbpfeats_%s_filenames' % (class_name), labels_or_filenames)

    # Cleanup. Better safe than sorry
    del features

    print('done extracting features')

    # Feature matrices were saved by class to save memory. We now
    # concatenate everything and prepare the label vector 
    print('STAGE 2: CLASS FEATURE CONCATENATION')

    all_feats = []
    labels = []

    for cidx, class_dir in enumerate(classes_dir):
        class_name = path.basename(class_dir)

        print('Loading class \"%s\" (%d of %d)' % (class_name, cidx + 1, len(classes_dir)))

        class_feats = np.load('../data/lbpfeats_%s.npy' % (class_name))

        all_feats.append(class_feats)

        if not is_test:
            labels += [class_name] * len(class_feats)
        else:
            labels = np.hstack((labels, np.load('../data/lbpfeats_%s_filenames.npy' % (class_name))))

    all_feats = np.vstack(all_feats)

    np.save('../data/512x512patch_rbw_lbp', all_feats)
    np.save('../data/512x512patch_rbw_lbp_labels', labels)

    print('Mischief managed.')

    return [all_feats, labels]

print('ready')
