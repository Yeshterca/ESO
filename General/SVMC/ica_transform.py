import numpy as np
from sklearn.decomposition import PCA
import scipy.io
import h5py
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def perform_pca_ica(subject, n_components, whitening_matrix, unmixing_matrix):
    # Reshape the data to 2D
    original_shape = subject.shape
    n_voxels = np.prod(original_shape[:-1])
    n_timepoints = original_shape[-1]
    reshaped_data = data.reshape(n_voxels, n_timepoints)

    # Center the data
    mean = np.mean(reshaped_data, axis=1, keepdims=True)
    centered_data = reshaped_data - mean

    centered_data = StandardScaler().fit_transform(centered_data)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(centered_data.T).T

    # Apply PCA whitening
    whitened_data = np.dot(whitening_matrix, pca_components)

    # Apply ICA unmixing
    ica_components = np.dot(unmixing_matrix, whitened_data)

    return ica_components


def temporal_regression(components, regressor):

    c_number = components.shape[0]
    betas = np.zeros(c_number)

    for i in range(0, c_number):
        c = components[i, :]
        model = LinearRegression().fit(regressor, c)
        betas[i] = model.coef_[0]

    return betas



data = scipy.io.loadmat(r'C:\Users\kajin\PhD\ESO\IKEM\ICA\ICA_apply\subject0.mat')
data = np.array(data['subject'])
data = data.astype('float64')

n_components = 36

unmixing_matrix = h5py.File(r'C:\Users\kajin\PhD\ESO\IKEM\ICA\ICA_apply\eso_ica.mat', 'r')
unmixing_matrix = np.array(unmixing_matrix['W'])
W = unmixing_matrix.astype('float64')

whitening_matrix = h5py.File(r'C:\Users\kajin\PhD\ESO\IKEM\ICA\ICA_apply\eso_icasso_results.mat', 'r')
whitening_matrix = np.array(whitening_matrix['sR/whiteningMatrix'])
whitening_matrix = whitening_matrix.astype('float64')

spm_mat = h5py.File(r'C:\Users\kajin\PhD\ESO\SPM.mat')
regressors = np.array(spm_mat['spmVar/xX/X'])
regressors = regressors.astype('float64')
regressor = regressors[0, :]

# Perform PCA and ICA on the original data
ica_components = perform_pca_ica(data, n_components, whitening_matrix, W)

# Fit linear regression model
betas = temporal_regression(ica_components, regressor.reshape(-1, 1))
print(betas)

