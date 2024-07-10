from ica_run import *

## SET VARIABLES FOR DATASETS, PERFORM CLASSIFICATION ON EACH DATASET SEPARATELY
dataset = 'ikem'
print(dataset)
no_subjects_ikem, no_hc_ikem, no_fes_ikem, relevant_ics_ikem, no_features_ica_ikem, directory_ica_ikem, path_ikem = set_vars(dataset)
features_ikem, labels_ikem, weights_ikem = ica_load_prep_data(directory_ica_ikem, no_subjects_ikem, no_hc_ikem, no_fes_ikem, relevant_ics_ikem, no_features_ica_ikem)
# acc_ica, sen_ica, spec_ica = ica_classify(features_ikem, labels_ikem, weights_ikem, 'linear')

# plot the distribution of beta values for selected ICs
hc_means, fes_means = betas_distribution(features_ikem,  relevant_ics_ikem, path_ikem, no_hc_ikem)
pos_mean_hc_ikem, neg_mean_hc_ikem, pos_mean_fes_ikem, neg_mean_fes_ikem = group_means(hc_means, fes_means, dataset)

dataset = 'nudz'
print('\n')
print(dataset)
no_subjects_nudz, no_hc_nudz, no_fes_nudz, relevant_ics_nudz, no_features_ica_nudz, directory_ica_nudz, path_nudz = set_vars(dataset)
features_nudz, labels_nudz, weights_nudz = ica_load_prep_data(directory_ica_nudz, no_subjects_nudz, no_hc_nudz, no_fes_nudz, relevant_ics_nudz, no_features_ica_nudz)
#acc_ica, sen_ica, spec_ica = ica_classify(features_nudz, labels_nudz, weights_nudz, 'linear')

# plot the distribution of beta values for selected ICs
hc_means, fes_means = betas_distribution(features_nudz,  relevant_ics_nudz, path_nudz, no_hc_nudz)
pos_mean_hc_nudz, neg_mean_hc_nudz, pos_mean_fes_nudz, neg_mean_fes_nudz = group_means(hc_means, fes_means, dataset)

p_hc, p_fes, n_hc, n_fes, pos, neg = group_diff(pos_mean_fes_ikem, neg_mean_fes_ikem, pos_mean_fes_nudz, neg_mean_fes_nudz, pos_mean_hc_ikem, neg_mean_hc_ikem, pos_mean_hc_nudz, neg_mean_hc_nudz)

# ## CLASSIFY NUDZ USING IKEM, ICA COMPUTED ON MERGED DATASETS
# no_subjects_nudz, no_hc_nudz, no_fes_nudz, relevant_ics_nudz, no_features_ica_nudz, directory_ica_nudz = set_vars('merged')
# ica_features, labels, weights = ica_load_prep_data(directory_ica_nudz, no_subjects_nudz, no_hc_nudz, no_fes_nudz, relevant_ics_nudz, no_features_ica_nudz)
#
# # shuffle subjects
# labels_s_ikem, weights_s_ikem, ica_features_s_ikem = shuffle_subjects(labels_ikem, weights_ikem, ica_features[0:no_subjects_ikem, :])
# labels_s_nudz, weights_s_nudz, ica_features_s_nudz = shuffle_subjects(labels_nudz, weights_nudz, ica_features[no_subjects_ikem:, :])
#
# print('\n')
# print('train on IKEM, classify NUDZ')
# accuracy, sensitivity, specificity = ica_classify_otherdataset(ica_features_s_ikem, labels_s_ikem, weights_s_ikem, ica_features_s_nudz, labels_s_nudz, weights_s_nudz, 'linear')


