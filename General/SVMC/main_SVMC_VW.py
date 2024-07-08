from vw_run import *
from plots_figures import plot_roc

# GLOBALS to tune
NO_FEATURES = 18
NO_SEGMENTS = 500
NO_SLICE = 55

# ONE DATASET LOO
# get variables
# no_subjects, no_hc, no_fes, directory = set_vars('nudz')
#
# # load data, get features, classify
# betas_LB, betas, betas_NB, labels, weights = vw_load_data(directory, 'beta_0001.nii', no_subjects, no_hc, no_fes)
# predictions, probs = vw_LOO_cluster_classify(betas_LB, betas, betas_NB, labels, weights, no_subjects, NO_SEGMENTS, NO_SLICE, NO_FEATURES)
#
# np.save('labelsf18_nudz', labels)
# np.save('probsf18_nudz.npy', probs)
# np.save('predsf18_nudz.npy', predictions)
#
# # labels = np.load('labelsv01.npy')
# # probs = np.load('probsv01.npy')
# # predictions = np.load('predsv01.npy')
#
# # get performance
# accuracy, sensitivity, specificity = get_measures(labels, predictions, weights)
# plot_roc(labels, probs)

# TWO DATASETS
# load data from both datasets
no_subjectsA, no_hcA, no_fesA, directoryA = set_vars('ikem')
no_subjectsB, no_hcB, no_fesB, directoryB = set_vars('nudz')

betas_largeA, betas_smallA, betas_nanA, labelsA, weightsA = vw_load_data(directoryA, 'beta_0001.nii', no_subjectsA, no_hcA, no_fesA)
betas_largeB, betas_smallB, betas_nanB, labelsB, weightsB = vw_load_data(directoryB, 'beta_0001.nii', no_subjectsB, no_hcB, no_fesB)

# normalization
betasA, mean, sd = z_score_normalize(betas_nanA, no_hcA)
betasA[np.isnan(betasA)] = 0.

betasB, mean, sd = z_score_normalize(betas_nanB, no_hcB)  # TODO !!!
betasB[np.isnan(betasB)] = 0.

# determine superpixels on dataset A
segments, _, _, _ = cluster(betas_largeA, betas_smallA, no_hcA, NO_SEGMENTS, NO_SLICE)

# compute superpixels means and find the transformation - using dataset A
plsA, pls_t = vw_features_find(betasA, labelsA, segments, NO_FEATURES, no_subjectsA)

# apply clustering on dataset B, compute superpixels means
pixel_meansB = pixel_means(betasB, segments, no_subjectsB)

# use found transformation and apply it to dataset B
plsB = pls_t.transform(pixel_meansB.T)

# classify B using A
predictions, probs = svm_classifier(plsA, labelsA, weightsA, plsB, kernel='linear')
accuracy, sensitivity, specificity = get_measures(labelsB, predictions, weightsB)
plot_roc(labelsB, probs)


