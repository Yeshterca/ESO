from nilearn import datasets, image, input_data, plotting
import nibabel as nib

# Fetch Craddock atlas, use temporal correlation
craddock_atlas = datasets.fetch_atlas_craddock_2012()['tcorr_mean']
craddock = image.index_img(craddock_atlas, 20)  # load slice with the result for 200 clusters

# Load fMRI data
fmri_img = image.load_img(r'C:\Users\kajin\PhD\swauJOYSTICK_30_3x3x3_20150713151754_3.nii')

# Create the masker object
masker = input_data.NiftiLabelsMasker(labels_img=craddock, standardize=True, memory='nilearn_cache')
plotting.plot_roi(craddock)

# Get time-series for each ROI
time_series = masker.fit_transform(r'C:\Users\kajin\PhD\swauJOYSTICK_30_3x3x3_20150713151754_3.nii')

print(time_series.shape)

