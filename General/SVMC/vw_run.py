from model import *
from vw_clustering import *


def vw_load_data(directory, filename, no_subjects, no_hc, no_fes):
    """
    Loads and prepares data, labels and weights.

    :param directory: full path to data directory
    :param filename: name of file with desired data (betas_001.nii)
    :param no_subjects: number of subjects
    :param no_hc: number of healthy
    :param no_fes: number of patients
    :return: betas_LB ... masked data with large background for clustering (10)
             betas... masked data with small backgroud (0)
             betas_NB ... masked data with nan values in the background (for normalization)
             [no_subjects, 91, 109, 91]
             labels, weights
    """

    betas_all = load_subjects(directory, filename, no_subjects)
    betas_LB, betas, betas_NB = masking(betas_all, no_subjects)
    labels = create_labels(no_hc, no_fes)
    weights = get_weights(labels)

    return betas_LB, betas, betas_NB, labels, weights

def vw_LOO_cluster_classify(betas_LB, betas, betas_NB, labels, weights, no_subjects, no_segments, no_slice, no_features):
    """
    Leave-one-out loop performs split to training and validation data, clustering, PLS projection and classification.

    :param betas_large: masked data with large background for clustering (10), [no_subjects, 91, 109, 91]
    :param betas_small: asked data with small backgroud (0), [no_subjects, 91, 109, 91] used for display
    :param labels: (no_subjects, ), 0/1
    :param weights: (no_subjects, ), weights to compensate for uneven dataset
    :param no_subjects: number of subjects
    :param no_segments: number of superpixels to find
    :param no_slice: number of slice to display
    :param no_features: number of features for classification
    :return: predictions ... (no_subjects, ), 0/1
             probs ... (no_subjects, ), probabilities from SVM
    """

    folds = len(labels)
    predictions = np.zeros((folds,))
    probs = np.zeros((folds,))

    for fold in range(folds):
        print('FOLD: ', fold)
        betas_train_LB, betas_train, betas_train_NB, labels_train, weights_train, betas_val = data_split(betas_LB,
                                                                                                  betas, betas_NB, labels,
                                                                                                  weights, fold)
        if fold <= 55:
            no_hc = 54
        else:
            no_hc = 55

        betas_train, mean, sd = z_score_normalize(betas_train_NB, no_hc)  # normalize training dataset
        betas_train[np.isnan(betas_train)] = 0.
        betas_val = (betas_val - mean)/sd
        betas_val[np.isnan(betas_val)] = 0  # normalize validation dataset

        # superpixels
        segments, overlay1, overlay2, overlay3 = cluster(betas_train_LB, betas_train, no_hc, no_segments, no_slice)

        # transformation - find and apply
        pls_train, pls = vw_features_find(betas_train, labels_train, segments, no_features, no_subjects-1)
        pls_val = vw_features_transform(pls, betas_val, segments)

        # classify
        prediction, prob = svm_classifier(pls_train, labels_train, weights_train, pls_val, kernel='linear')
        predictions[fold] = prediction[0]
        probs[fold] = prob[0]

    return predictions, probs


def data_split(betas_LB, betas, betas_NB, labels, weights, fold):
    """
    Split data to training and validation.

    :param betas_large: masked data with large background for clustering (10), [no_subjects, 91, 109, 91]
    :param betas: masked data with small backgroud (0), [no_subjects, 91, 109, 91]
    :param betas_NB: masked data with nan values in the background, [no_subjects, 91, 109, 91]
    :param labels: (no_subjects, ), 0/1
    :param weights: (no_subjects, ), weights to compensate for uneven dataset
    :param fold: "fold number"
    :return: betas_train_LB ... training data, large background [no_subjects, 91, 109, 91]
             betas_train ... training data, small background [no_subjectS, 91, 109, 91]
             betas_train_NB ... training data, nan values in the background [no_subjects, 91, 109, 91]
             labels_train ... training labels (no_subjects, )
             weights_train ... training weights (no_subjects, )
             betas_val ... validation data [1, 91, 109, 91]
    """

    betas_train_LB = np.delete(betas_LB, fold, 0)
    betas_train = np.delete(betas, fold, 0)
    betas_train_NB = np.delete(betas_NB, fold, 0)

    labels_train = np.delete(labels, fold)
    weights_train = np.delete(weights, fold)

    betas_val = betas_NB[fold, :, :, :]  # normally in LOO ok  #todo nb
    #betas_val = betas_small[fold, :]  # in case of features opt.

    return betas_train_LB, betas_train, betas_train_NB, labels_train, weights_train, betas_val


def cluster(betas_train_large, betas_train_small, no_hc, no_segments, no_slice):
    """
    Cluster data using SLIC superpixels.

    :param betas_train_large: training data, large background [no_subjects-1, 91, 109, 91]
    :param betas_train_small: training data, small background [no_subjects-1, 91, 109, 91]
    :param no_hc: number of healthy
    :param no_segments: number of superpixels for clustering
    :param no_slice: number of slice to display
    :return: segments ... clustered data (labels)
             overlay1 ... slice to display clustering, transversal
             overlay2 ... frontal
             overlay3 ... sagital
    """

    betas_conc_large = mean_groups_for_superpixels(betas_train_large, no_hc)
    betas_conc_small = mean_groups_for_superpixels(betas_train_small, no_hc)
    segments, overlay1, overlay2, overlay3 = superpixels(betas_conc_large, betas_conc_small, no_segments, no_slice)

    return segments, overlay1, overlay2, overlay3


def vw_features_find(betas_train_small, labels_train, segments, no_features, no_subjects):
    """
    Find PLS transformation and transform training data.

    :param betas_train_small: training data, small background [no_subjects-1, 91, 109, 91]
    :param labels_train: training labels (no_subjects-1, )
    :param segments: clustered data (labels)
    :param no_features: number of features for SVM
    :param no_subjects: number of subjects
    :return: pls_train ... pls tranformation fitted to training data
             pls ... pls transformation learned on training data
             pixel_means_all ... means of superpixels
    """

    pixel_means_train = pixel_means(betas_train_small, segments, no_subjects)  # -1 ... in LOO
    pls_train, pls = get_features_pls(pixel_means_train.T, labels_train, no_features)

    return pls_train, pls


def vw_features_transform(pls, betas_val, segments):
    """
    Transform validation data using PLS transformation found on training data.

    :param pls: transformation
    :param betas_val: validation data [1, 91, 109, 91]
    :param segments: clustered data (labels)
    :return: pls_val ... transformed validation data
    """

    clusted_subject = betas_in_superpixels(betas_val, segments)
    pixel_means_val = mean_superpixels(clusted_subject)

    pixel_means_val = pixel_means_val.reshape(1, -1)
    pls_val = pls.transform(pixel_means_val)

    return pls_val


def set_vars(dataset):
    """
    Set variables of IKEM and NUDZ dataset.

    :param dataset: ikem / nudz, string
    :return: no_subjects ... number of subjects in the dataset
             no_hc ... number of HC
             no_fes ... number of FES
             directory_vw ... directory with the data
    """
    if dataset == 'ikem':
        # GLOBALS IKEM
        no_subjects = 131
        no_hc = 55
        no_fes = 76
        directory_vw = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\VOXEL-WISE\1st_level_fovselected5'
        directory_figures = r'C:\Users\kajin\PhD\ESO\IKEM\VW\betas_distributions'

    elif dataset == 'nudz':
        # GLOBALS NUDZ
        no_subjects = 158
        no_hc = 66
        no_fes = 92
        directory_vw = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\data_nudz\NUDZ_vw\1st_level'
        directory_figures = r'C:\Users\kajin\PhD\ESO\NUDZ\VW\betas_distributions'

    return no_subjects, no_hc, no_fes, directory_vw, directory_figures










