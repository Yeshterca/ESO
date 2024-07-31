from nilearn import datasets, image, input_data, plotting
import nibabel as nib
import os
import numpy as np


def load_atlas(clusters_slice, parc_type):

    atlas = datasets.fetch_atlas_craddock_2012()[parc_type]
    parcellation = image.index_img(atlas, clusters_slice)

    masker = input_data.NiftiLabelsMasker(labels_img=parcellation, standardize=True, memory='nilearn_cache')
    #plotting.plot_roi(parcellation)

    return masker


def get_rois(masker, dirlist):

    i = 0
    with open(dirlist, 'r') as file:
        subjects = file.readlines()
        no_subjects = len(subjects)
        for file in subjects:
            file = file.strip()
            file = r'{}'.format(file)
            if i == 0:
                time_series = masker.fit_transform(file)
            else:
                time_series_all = np.zeros((time_series.shape[0], time_series.shape[1], no_subjects))
                time_series_all[:, :, 0] = time_series
                time_series_all[:, :, i] = masker.fit_transform(file)
            i += 1

    return time_series_all


dirlist = r'C:\Users\kajin\PhD\zkouska.txt'

masker = load_atlas(20, 'tcorr_mean')
#time_series_all = masker.fit_transform(r'C:\Users\kajin\PhD\swauJOYSTICK_30_3x3x3_20150713151754_3.nii')
time_series = get_rois(masker, dirlist)


