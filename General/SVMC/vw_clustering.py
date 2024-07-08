import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn.cross_decomposition import PLSRegression
import os
import nibabel as nib


def load_subjects(dirpath, filename, no_subjects):
    """
    Load the data in a form of [91, 109, 91] sized matrix of beta values for each subject.

    :param dirpath: path to the directory with the data
    :param filename: data matrix file name
    :param no_subjects: number of subjects
    :return: betas_all ... [no_subjects, 91, 109, 91] matrix
    """

    betas_all = np.zeros((no_subjects, 91, 109, 91))
    i = 0
    for dire in os.listdir(dirpath):
        for file in os.listdir(dirpath + '\\' + dire):
            if file == filename:
                sub_dir = os.path.join(dirpath, dire, filename)
                betas_subject = nib.load(sub_dir).get_fdata()
                betas_all[i, :, :, :] = betas_subject
        i += 1

    return betas_all

def masking(betas_all, no_subjects):
    """
    Create a mask using all subjects - where any subject has a nan value - nan/0/10 in a mask.

    :param betas_all: [no_subjects, 91, 109, 91] data matrix comprising all subjects
    :param no_subjects: number of subjects
    :return: betas_LB ... data matrix, background voxels have large values
             betas ... data matrix, background voxels have 0 values
             betas_NB ... data matrix, background voxels have NaN values
    """

    mask = np.ones_like(betas_all[0, :, :, :])
    count_nans = np.zeros_like(betas_all[0, :, :, :])

    for i in range(0, no_subjects):
        nan_mask = np.isnan(betas_all[i, :, :, :])
        count_nans = count_nans + nan_mask
    mask[count_nans > 0] = 0

    betas_nan0 = betas_all
    betas_nanout = np.isnan(betas_all)
    betas_nan0[betas_nanout] = 0

    betas = np.zeros_like(betas_all)
    for i in range(0, no_subjects):
        betas[i, :, :, :] = betas_nan0[i, :, :, :] * mask

    betas_LB = np.where(betas == 0., 10., betas)
    betas_NB = np.where(betas == 0., np.nan, betas)

    return betas_LB, betas, betas_NB


def create_labels(no_hc, no_fes):
    """
    Create labels for data.

    :param no_hc: number of healthy
    :param no_fes: number of patients
    :return: labels (no_subjects,)
    """

    hc_labels = np.zeros((1, no_hc), 'int')
    fes_labels = np.ones((1, no_fes), 'int')
    labels = np.squeeze(np.concatenate((hc_labels, fes_labels), axis=1))

    return labels


def get_weights(labels):
    """
    Create weights for data to compensate for uneven dataset.

    :param labels: 0/1 (no_subjects, )
    :return: weights (no_subjects, )
    """

    size = len(labels)

    fes = np.where(labels == 1)[0]
    hc = np.where(labels == 0)[0]

    w_fes = (1. / len(fes)) * (size / 2)
    w_hc = (1. / len(hc)) * (size / 2)

    weights = np.empty(size)
    weights[fes] = w_fes
    weights[hc] = w_hc

    return weights


def pixel_means(betas_all, segments, no_subjects):
    """
    Compute superpixel means.

    :param betas_all: data to label with superpixels
    :param segments: clustered data (labels)
    :param no_subjects: number of subjects
    :return: pixel_means ... superpixel means
    """

    for i in range(0, no_subjects):
        betas_subj_clustered = betas_in_superpixels(betas_all[i, :, :, :], segments)
        if i == 0:
            pixel_means = np.zeros((len(betas_subj_clustered), no_subjects))
        pixel_means[:, i] = mean_superpixels(betas_subj_clustered)

    return pixel_means


def superpixels(betas, betas_for_disp, num_segments, slice_n):
    """
    Compute superpixels using SLIC.

    :param betas: data (no_subjects, 91, 109, 91) with large background (10)
    :param betas_for_disp: data with small background (0)
    :param num_segments: number of superpixels to cluster
    :param slice_n: number of slice to display
    :return: segments ... clustered data (labels)
             overlay1 ... slice to display clustering, transversal
             overlay2 ... frontal
             overlay3 ... sagital
    """

    segments = slic(betas, num_segments, compactness=0.2, channel_axis=3, enforce_connectivity=True)

    overlay1 = mark_boundaries(betas_for_disp[:, :, slice_n, 0], segments[:, :, slice_n])
    overlay2 = mark_boundaries(betas_for_disp[slice_n, :, :, 0], segments[slice_n, :, :])
    overlay3 = mark_boundaries(betas_for_disp[:, slice_n, :, 0], segments[:, slice_n, :])

    return segments, overlay1, overlay2, overlay3


def betas_in_superpixels(betas, segments):
    """
    Assign pixels to superpixel labels.

    :param betas: data
    :param segments: clustered data (labels)
    :return: betas_sup ... assigned
    """

    betas_sup = [[] for _ in range(np.max(segments)+1)]
    for (x, y, z), label in np.ndenumerate(segments):
        betas_sup[label].append(betas[x, y, z])
    betas_sup = [np.array(vals) for vals in betas_sup]

    return betas_sup


def extract_feature(superpixel_values):
    """
    Compute mean of one superpixel, ignore nans.

    :param superpixel_values: betas in one superpixel
    :return: mean
    """

    return np.nanmean(superpixel_values)


def mean_superpixels(betas_superpixels):
    """
    Compute mean of each superpixel.

    :param betas_superpixels: betas assigned to superpixels
    :return: pixel_means ... mean values of each superpixel
    """

    pixel_means = np.zeros(len(betas_superpixels))
    for i in range(len(betas_superpixels)):
        if len(betas_superpixels[i]) == 0:
            continue
        else:
            np.where(betas_superpixels[i] == 0., np.nan, betas_superpixels[i])
            pixel_means[i] = extract_feature(betas_superpixels[i])
    return pixel_means


def get_features_pls(pixel_means, labels_train, no_features):
    """
    Find PLS transformation, transform training data.

    :param pixel_means: means of superpixels (no_segments,)
    :param labels_train: labels of training data (no_subjects-1, )
    :param no_features: number of features for SVM
    :return: transform_train ... transformed training data
             pls ... PLS transformation
    """

    pls = PLSRegression(no_features)
    pls.fit(pixel_means, labels_train)
    transform_train = pls.transform(pixel_means)

    return transform_train, pls


def mean_groups_for_superpixels(betas_all, no_hc):
    """
    Create matrix where last dim are means over group of HC and FES.

    :param betas_all: data (no_subjects, 91, 109, 91)
    :param no_hc: number of healthy
    :return: betas_conc ... concatenated matrix of means of group (91, 109, 91, 2)
    """

    # mean of groups
    betas_HC_one = np.mean(betas_all[0:no_hc, :, :, :], axis=0)
    betas_FES_one = np.mean(betas_all[no_hc:, :, :, :], axis=0)

    # create one matrix (two channels)
    betas_conc = np.stack((betas_HC_one, betas_FES_one), axis=3)

    return betas_conc


def z_score_normalize(data, no_hc):
    """
    Normalize the data (two groups).

    :param data: data to normalize, consists of two groups
    :param no_hc: number of HC
    :return: normalized_data
             mean ... overall mean
             sd ... overall standard deviation
    """

    data_hc = data[0:no_hc, :, :, :]
    data_fes = data[no_hc:, :, :, :]

    mean_hc = np.nanmean(data_hc)
    sd_hc = np.nanstd(data_hc)

    mean_fes = np.nanmean(data_fes)
    sd_fes = np.nanstd(data_fes)

    mean = np.nanmean((mean_hc, mean_fes))
    sd = np.nanmean((sd_hc, sd_fes))

    normalized_data = (data-mean)/sd

    return normalized_data, mean, sd


