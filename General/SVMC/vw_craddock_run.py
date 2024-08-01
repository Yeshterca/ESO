from nilearn import datasets, image, input_data, plotting
import numpy as np
from sklearn.linear_model import LinearRegression
import h5py
from get_data_ICA import create_labels
from model import get_weights, get_measures, svm_classifier
from plots_figures import plot_roc
from sklearn.cross_decomposition import PLSRegression


def set_vars(dataset):

    if dataset == 'ikem':
        no_subjects = 131
        no_hc = 55
        no_fes = 76
        no_features = 10
        clusters_slice = 10
        subjects_list = r'/home/karolina.volfikova/Documents/ESO/datalists/IKEM/subs_dirs_fov5.txt'

    spm_dir = r'/home/karolina.volfikova/Documents/ESO/SPM_matrix/ICA_240/SPM.mat'
    atlas_dir = r"/home/karolina.volfikova/Documents/Craddock/tcorr05_mean_all.nii"

    return no_subjects, no_hc, no_fes, no_features, clusters_slice, subjects_list, spm_dir, atlas_dir


def cluster_regress(clusters_slice, atlas_dir, subjects_list, spm_dir):

    SPM_mat = h5py.File(spm_dir, 'r')
    time_experiment = SPM_mat['spmVar/xX/X'][0, :]

    masker = load_atlas(clusters_slice, atlas_dir)
    time_series_all = get_rois(masker, subjects_list)  # get time-series for each cluster
    betas = get_betas(time_series_all, time_experiment)  # get beta value for each cluster

    return betas


def load_atlas(clusters_slice, atlas_dir):

    parcellation = image.index_img(atlas_dir, clusters_slice)
    masker = input_data.NiftiLabelsMasker(labels_img=parcellation, standardize=True, memory='nilearn_cache')

    return masker


def get_rois(masker, dirlist):
    print('--- Clustering with Craddock---')
    i = 0
    with open(dirlist, 'r') as file:
        subjects = file.readlines()
        no_subjects = len(subjects)
        for file in subjects:
            print('subject ', i + 1)
            file = file.strip()
            file = r'{}'.format(file)
            if i == 0:
                time_series = masker.fit_transform(file)
                time_series_all = np.zeros((time_series.shape[0], time_series.shape[1], no_subjects))
                time_series_all[:, :, 0] = time_series
                print(np.shape(time_series_all))
            else:
                time_series_all[:, :, i] = masker.fit_transform(file)
            i += 1

    return time_series_all


def compute_regression(time_experiment, time_subject):
    model = LinearRegression()
    model.fit(time_experiment, time_subject)

    coefficients = model.coef_

    return coefficients


def get_betas(time_series, time_experiment):
    print('--- Linear regression ---')
    no_subjects = np.shape(time_series)[2]
    no_clusters = np.shape(time_series)[1]

    coeffs = np.zeros((no_subjects, no_clusters))
    for i in range(no_subjects):
        print('subject ', i + 1)
        for j in range(no_clusters):
            coeffs[i, j] = compute_regression(time_experiment.reshape(-1, 1), time_series[:, j, i])  # single feature
            # experiment = X, activity = y

    return coeffs

def load_prep_data(no_hc, no_fes):

    labels = create_labels(no_hc, no_fes)
    weights = get_weights(labels)

    # TODO load betas form a file

    return labels, weights


def vw_classify_craddock(features, labels, weights, kernel, no_features):

    predictions, probs = vw_LOOCV(features, labels, weights, kernel, no_features)

    print('--- VW-CLASSIFICATION ---')
    accuracy, sensitivity, specificity = get_measures(labels, predictions, weights)
    plot_roc(labels, probs)

    return accuracy, sensitivity, specificity


def vw_LOOCV(betas, labels, weights, kernel, no_features):

    folds = len(labels)
    predictions = np.zeros((folds,))
    probs = np.zeros((folds,))

    for fold in range(folds):
        # Split the data
        betas_train, labels_train, weights_train, betas_val = data_split(betas, labels, weights, fold)

        # Normalize
        # TODO

        # Perform PLS
        features_train, features_val = pls(betas_train, labels_train, betas_val, no_features)

        # Classify
        prediction, prob = svm_classifier(features_train, labels_train, weights_train, features_val, kernel)
        predictions[fold] = prediction[0]
        probs[fold] = prob

    return predictions, probs


def data_split(features, labels, weights, index):

    features_train = np.delete(features, index, 0)  # TODO check axis
    labels_train = np.delete(labels, index)
    weights_train = np.delete(weights, index)

    features_val = features[index]
    features_val = features_val.reshape(1, -1)

    return features_train, labels_train, weights_train, features_val


def pls(betas_train, labels_train, betas_val, no_features):

    # Find PLS transformation of the training data
    pls = PLSRegression(no_features)
    pls.fit(betas_train, labels_train)
    features_train = pls.transform(betas_train)

    # Use the transformation on the validation data
    features_val = pls.transform(betas_val)

    return features_train, features_val


