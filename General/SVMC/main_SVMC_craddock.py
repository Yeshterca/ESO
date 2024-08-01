from craddock_run import *

# Get data
no_subjects, no_hc, no_fes, no_features, clusters_slice, subjects_list, spm_dir, atlas_dir = set_vars('ikem')
betas = cluster_regress(clusters_slice, atlas_dir, subjects_list, spm_dir)
np.save('betas_craddock.npy', betas)

# TODO: save and load betas from a file
labels, weights = load_prep_data(no_hc, no_fes)

# Perform classification, LOOCV
accuracy, sensitivity, speicificity = classify_craddock(betas, labels, weights, 'linear', no_features, no_hc)


