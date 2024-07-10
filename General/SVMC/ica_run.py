from get_data_ICA import *
from model import *
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from plots_figures import *


# LOAD AND PREPARE DATA FROM ICA
def ica_load_prep_data(directory, no_subjects, no_hc, no_fes, relevant_ics, no_components):
    """
    Selects components of given numbers, creates labels and weights for the dataset, performs normalization.
    """

    # load all
    component_numbers, betas_all = get_ica_data(directory)
    # select significant ics
    ic_numbers_selected, ic_betas_selected = select_components(no_subjects, component_numbers, relevant_ics, betas_all, no_components)

    # get labels and weights
    labels = create_labels(no_hc, no_fes)
    weights = get_weights(labels)

    # normalization
    features_ica = z_score_normalize(ic_betas_selected.T, no_hc)

    return features_ica, labels, weights


# SET VARIABLES BASED ON THE DATASET
def set_vars(dataset):
    if dataset == 'ikem':
        # GLOBALS IKEM
        no_subjects = 131
        no_hc = 55
        no_fes = 76
        relevant_ics = np.array([5, 7, 15, 16, 19, 20, 22, 24, 30, 31])
        no_features_ica = len(relevant_ics)
        directory_ica = r'C:\Users\kajin\PhD\ESO\IKEM\ICA\eso_temporal_regression.mat'
        path_figures = r'C:\Users\kajin\PhD\ESO\IKEM\ICA\betas_distributions'

    elif dataset == 'nudz':
        # GLOBALS NUDZ only
        no_subjects = 158
        no_hc = 66
        no_fes = 92
        relevant_ics = np.array([2, 3, 5, 13, 18, 22, 23, 30])
        no_features_ica = len(relevant_ics)
        directory_ica = r'C:\Users\kajin\PhD\ESO\NUDZ\ICA\model_performance\NUDZ_only\eso_temporal_regression.mat'
        path_figures = r'C:\Users\kajin\PhD\ESO\NUDZ\ICA\betas_distributions'

    elif dataset == 'merged':
        # GLOBALS IKEM + NUDZ
        no_subjects = 289
        no_hc = 121
        no_fes = 168
        relevant_ics = np.array([5, 6, 8, 17, 18, 20, 21, 22, 26, 31, 34])
        no_features_ica = len(relevant_ics)
        directory_ica = r'C:\Users\kajin\Desktop\PhD\ESO\NUDZ\ICA\IKEM_NUDZ_merged\eso_temporal_regression.mat'

    return no_subjects, no_hc, no_fes, relevant_ics, no_features_ica, directory_ica, path_figures


def shuffle_subjects(labels, weights, features):

    r = np.array(range(labels.shape[0]))
    np.random.shuffle(r)
    labels_s = labels[r]
    weights_s = weights[r]
    features_s = features[r, :]

    return labels_s, weights_s, features_s


# CLASSIFY
def ica_classify(betas, labels, weights, kernel):
    """
    Performs leave-one-out cross validation and gets the performance of the model.
    """

    predictions, probs = ica_LOO_cross_validation(betas, labels, weights, kernel)

    print('---------- ICA-CLASSIFICATION ----------')
    accuracy, sensitivity, specificity = get_measures(labels, predictions, weights)
    plot_roc(labels, probs)

    return accuracy, sensitivity, specificity

def ica_classify_otherdataset(ica_features_ikem, labels_ikem, weights_ikem, ica_features_nudz, labels_nudz, weights_nudz, kernel):

    predictions, probs = svm_classifier(ica_features_ikem, labels_ikem, weights_ikem, ica_features_nudz, kernel)

    print('---------- ICA-CLASSIFICATION ----------')
    accuracy, sensitivity, specificity = get_measures(labels_nudz, predictions, weights_nudz)
    plot_roc(labels_nudz, probs)

    return accuracy, sensitivity, specificity

