from nilearn import datasets, image, input_data, plotting
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import h5py


def load_atlas(clusters_slice, parc_type):

    #atlas = datasets.fetch_atlas_craddock_2012()[parc_type]
        # url=r"C:\Users\kajin\PhD\Craddock\craddock_2011_parcellations.tar.gz"
    atlas = r"C:\Users\kajin\PhD\Craddock\craddock_2011_parcellations\tcorr05_mean_all.nii\tcorr05_mean_all.nii"
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
                time_series_all = np.zeros((time_series.shape[0], time_series.shape[1], no_subjects))
                time_series_all[:, :, 0] = time_series
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

    no_subjects = np.shape(time_series)[2]
    no_clusters = np.shape(time_series)[1]

    coeffs = np.zeros((no_subjects, no_clusters))
    for i in range(no_subjects):
        for j in range(no_clusters):
            coeffs[i, j] = compute_regression(time_experiment.reshape(-1, 1), time_series[:, j, i])  # single feature
                # experiment = X, activity = y

    return coeffs


def perform_pca(coeffs, no_components):

    #data = StandardScaler().fit_transform(coeffs)

    pca = PCA(no_components)
    components = pca.fit_transform(coeffs)

    return components


dirlist = r'C:\Users\kajin\PhD\zkouska.txt'
SPM_mat = h5py.File(r'C:\Users\kajin\PhD\ESO\SPM.mat', 'r')
time_experiment = SPM_mat['spmVar/xX/X'][0, :]

masker = load_atlas(20, 'tcorr_mean')
#time_series_all = masker.fit_transform(r'C:\Users\kajin\PhD\swauJOYSTICK_30_3x3x3_20150713151754_3.nii')
time_series = get_rois(masker, dirlist)

coeffs = get_betas(time_series, time_experiment)
print(np.shape(coeffs))

components = perform_pca(coeffs, 3)

print(components)

# TODO perform z-score normalization of the data


