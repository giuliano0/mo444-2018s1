#!/usr/bin/env python3

import numpy as np
import glob
import os

from os import path

from scipy import stats
from scipy.signal import correlate2d

from skimage.io import imread
from skimage.color import rgb2ycbcr
from skimage.restoration import denoise_wavelet

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

    # Possible change of parameters: multichannel to True, wavelet to other type
    denoised_patch = denoise_wavelet(patch, wavelet='db5', convert2ycbcr=False, multichannel=True)

    return patch.astype(np.float64) - denoised_patch


# Do not use
def extract_features(noise_plane):
    '''
    The input is a plane array with a noise pattern.
    Output is a feature vector based on that pattern.
    '''
    desc = stats.describe(noise_plane, axis=None)
    _, _, mean, var, skew, kurt = desc

    feats = np.array([mean, var, skew, kurt])

    return feats


def process_image_subroutine(img_path):
    print(' Starting image %s' % (img_path), end='\r')

    img = imread(img_path, plugin='matplotlib').astype(np.float64)
    patch = extract_patch(img)
    patch = append_luminance(patch)
    noises = get_noise(patch)

    return noises # one array (psz, psz, 4)


def extract_noise_patterns(class_dir):
    '''
    Returns a list with as many elements as there are images in class_dir.
    Each element is a (512, 512, 4) numpy.ndarray of float64 containing
    each image's estimated SPN.
    '''
    ref_spn = []                    # reference noise pattern, 1 per class
    #spns_labels = []
    noise_patterns = []             # noise patterns, 1 per image
    noise_patterns_labels = []

    image_paths = glob.glob(path.join(class_dir, '*.[Jj][Pp][Gg]')) # MFing case sensitiveness
    class_name = path.basename(class_dir)

    proc_pool = Pool(4)

    # Let's try this in parallel
    noise_patterns = proc_pool.map(process_image_subroutine, image_paths)

    # Close pool and wait for task completion
    proc_pool.close()
    proc_pool.join()

    print('') # prints nothing but adds a \n to the end of the line

    # noise_pattern is a list: (n images, 1 patch, 4 channels)
    # now calculate a mean across the image dimension
    ref_spn = np.mean(noise_patterns, axis=0)

    noise_patterns_labels += [class_name] * len(noise_patterns)
    
    return [ref_spn, class_name, noise_patterns, noise_patterns_labels]


def get_correlations(refs, noise):
    '''
    ref: reference SPN for all classes (512, 512, 4, 10)
    noise: SPN for a given image (512, 512, 4)
    
    Returns: (40,), i.e., 40 correlations by image
    '''
    _, _, d = noise.shape

    feats = []

    for ref in refs:
        for i in range(d):
            pc, _ = stats.pearsonr(noise[:, :, i].ravel(), ref[:, :, i].ravel())
            feats += [pc]

    return feats


def concatenate_ref_spns(classes_dirs):
    all_ref_spn = []

    for class_dir in classes_dirs:
        class_name = path.basename(class_dir)
        ref_spn = np.load('../data/ref_spn_%s.npy' % (class_name))

        all_ref_spn.append(ref_spn)
    
    return all_ref_spn


def the_algorithm():
    #DATASET_DIR = '../dataset/train'
    DATASET_DIR = '../problematic-motherfuckers'

    # Let's extract for each class
    # Then save the SPN reference (1 guy per class) and each image SPN (m images per class)
    # Then we load 1 SPN reference and load each class SPNs and calculate correlations, save results
    # Repeat for all SPN references
    # Open each result file, concatenate
    # Profit

    # Each directory inside dataset/train/ is a class
    classes_dir = glob.glob(path.join(DATASET_DIR, '*'))

    print('STAGE 1: SPN EXTRACTION (CLASS REFERENCE AND ALL IMAGES)')

    for cidx, class_dir in enumerate(classes_dir):
        class_name = path.basename(class_dir)

        print('Starting class \"%s\" (%d of %d)' % (class_name, cidx + 1, len(classes_dir)))
        print('class_dir %s' % (class_dir))

        ref_spn, _, noise_patterns, _ = extract_noise_patterns(class_dir)

        # No need to save labels. I have them by the class_name variable
        np.save('../data/ref_spn_%s' % (class_name), ref_spn)
        np.save('../data/spn_%s' % (class_name), noise_patterns)

    # Cleanup. Better safe than sorry
    del ref_spn, noise_patterns

    # Let's concatenate all reference SPNs
    all_ref_spns = concatenate_ref_spns(classes_dir)

    print('STAGE 2: CORRELATION CALCULATION')

    all_corrs = []
    labels = []

    for cidx, class_dir in enumerate(classes_dir):
        class_name = path.basename(class_dir)
        images_spn = np.load('../data/spn_%s.npy' % (class_name))

        print('Starting class \"%s\" (%d of %d)' % (class_name, cidx + 1, len(classes_dir)))

        for image_spn in images_spn:
            all_corrs.append(get_correlations(all_ref_spns, image_spn))
        
        labels += [class_name] * len(images_spn)

        # At this point, all_corrs should be a (M, 40) array, where M
        # is the number of images, containing all correlations between
        # a reference SPN from class c1 against all SPNs of images of
        # class c2

        np.save('../data/correlations', all_corrs)

    print('done extracting features')

    return [np.matrix(all_corrs), labels]

print('done')
