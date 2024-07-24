import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import os

# ROIS HISTOGRAM

def plot_rois_distribution(dir):

    rois_out = np.load(dir)
    rois_out = np.squeeze(rois_out)

    n, bins, patches = plt.hist(x=rois_out, bins=16, color='darkblue',
                                alpha=0.7, rwidth=0.85, label='subjects')

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.xticks(bin_centers, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    plt.vlines(14, 0, 21, colors='red', linestyles='dashed', label='threshold')  #26.1, 48
    plt.xlabel('Number of ROIs')
    plt.ylabel('Number of subjects')
    plt.title('Histogram of ROIs less than 50% covered in subjects, NUDZ')
    plt.legend()
    plt.show()


def plot_features_accuracy(accuracies, no_features):


    fig = plt.figure()
    plt.plot(no_features, accuracies, color='darkblue')
    plt.xlabel('Number of features')
    plt.ylabel('Weighted accuracy')
    plt.title('Accuracy of the model in relation to the number of features')
    plt.grid(which='major')
    plt.show()


def display_clustering(overlay1, overlay2, overlay3):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(overlay1[:, :, 0])
    ax2.imshow(overlay2[:, :, 0])
    ax3.imshow(overlay3[:, :, 0])

    fig.suptitle('OVERLAYS')
    plt.show()


def display_mean_groups(betas_conc, slice):

    fig = plt.figure()
    axs = fig.subplot_mosaic([["HC 1", "FES 1"],
                              ["HC 2", "FES 2"],
                              ["HC 3", "FES 3"]])
    axs["HC 1"].set_title("HC 1")
    axs["HC 1"].imshow(betas_conc[slice, :, :, 0])
    axs["FES 1"].set_title("FES 1")
    axs["FES 1"].imshow(betas_conc[slice, :, :, 1])

    axs["HC 2"].set_title("HC 2")
    axs["HC 2"].imshow(betas_conc[:, slice, :, 0])
    axs["FES 2"].set_title("FES 2")
    axs["FES 2"].imshow(betas_conc[:, slice, :, 1])

    axs["HC 3"].set_title("HC 3")
    axs["HC 3"].imshow(betas_conc[:, :, slice, 0])
    axs["FES 3"].set_title("FES 3")
    axs["FES 3"].imshow(betas_conc[:, :, slice, 1])

    plt.show()


def display_subjects(betas, number, slice=55):
    plt.imshow(betas[:, slice, :])
    plt.savefig(number)


def plot_roc(labels, probs):

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='cornflowerblue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('VW: Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def betas_distributions_c(betas, c_numbers, path, no_hc):

    hc_means = np.zeros(len(c_numbers), )
    fes_means = np.zeros(len(c_numbers), )
    for i in range(len(c_numbers)):
        hc = betas[0:no_hc, i]
        fes = betas[no_hc+1:, i]
        min_edge = min(np.min(hc), np.min(fes))
        max_edge = max(np.max(hc), np.max(fes))
        bins = np.linspace(min_edge, max_edge, 20)

        n_hc, bins_hc, _ = plt.hist(x=hc, bins=bins, color='red', alpha=0.5, rwidth=0.85, label='HC')
        n_fes, bins_fes, _ = plt.hist(x=fes, bins=bins, color='darkblue', alpha=0.5, rwidth=0.85, label='FES')

        y_max = np.max([np.max(n_hc), np.max(n_fes)]) + 1
        hc_means[i] = np.mean(hc)
        fes_means[i] = np.mean(fes)

        plt.vlines(x=np.mean(hc), ymin=0, ymax=y_max,  colors='red', linestyles='dashed', label='HC-mean')
        plt.vlines(x=np.mean(fes), ymin=0, ymax=y_max, colors='darkblue', linestyles='dashed', label='FES-mean')
        plt.xlabel('Beta values')
        plt.ylabel('Number of subjects')
        plt.title(f'Distribution of beta values, C: {c_numbers[i]}')
        plt.legend()
        plt.savefig(os.path.join(path, f'betas_distribution_C{c_numbers[i]}'))
        plt.show()

    return hc_means, fes_means


def betas_distribution_c(betas, c_numbers, path, no_hc):

    betas_hc = (betas[0:no_hc, :]).flatten()
    betas_fes = (betas[no_hc+1:, :]).flatten()

    min_edge = min(np.min(betas_hc), np.min(betas_fes))
    max_edge = max(np.max(betas_hc), np.max(betas_fes))
    bins = np.linspace(min_edge, max_edge, 20)

    n_hc, bins_hc, _ = plt.hist(x=betas_hc, bins=bins, color='red', alpha=0.5, rwidth=0.85, label='HC')
    n_fes, bins_fes, _ = plt.hist(x=betas_fes, bins=bins, color='darkblue', alpha=0.5, rwidth=0.85, label='FES')

    y_max = np.max([np.max(n_hc), np.max(n_fes)]) + 1

    plt.vlines(x=np.mean(betas_hc), ymin=0, ymax=y_max,  colors='red', linestyles='dashed', label='HC-mean')
    plt.vlines(x=np.mean(betas_fes), ymin=0, ymax=y_max, colors='darkblue', linestyles='dashed', label='FES-mean')
    plt.xlabel('Beta values')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of beta values, all C')
    plt.legend()
    plt.savefig(os.path.join(path, f'betas_distribution_allC'))
    plt.show()

def betas_distribution_vw(betas, no_hc, path):

    betas_hc = betas[0:no_hc, :, :, :].flatten()
    betas_fes = betas[no_hc+1:, :, :, :].flatten()

    betas_hc = betas_hc[~np.isnan(betas_hc)]
    betas_fes = betas_fes[~np.isnan(betas_fes)]

    min_edge = min(np.min(betas_hc), np.min(betas_fes))
    max_edge = max(np.max(betas_hc), np.max(betas_fes))
    bins = np.linspace(min_edge, max_edge, 20)

    n_hc, bins_hc, _ = plt.hist(x=betas_hc, bins=bins, color='red', alpha=0.5, rwidth=0.85, label='HC')
    n_fes, bins_fes, _ = plt.hist(x=betas_fes, bins=bins, color='darkblue', alpha=0.5, rwidth=0.85, label='FES')

    y_max = np.max([np.max(n_hc), np.max(n_fes)]) + 1

    plt.vlines(x=np.mean(betas_hc), ymin=0, ymax=y_max,  colors='red', linestyles='dashed', label='HC-mean')
    plt.vlines(x=np.mean(betas_fes), ymin=0, ymax=y_max, colors='darkblue', linestyles='dashed', label='FES-mean')
    plt.xlabel('Beta values')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of beta values')
    plt.legend()
    plt.savefig(os.path.join(path, f'betas_distribution'))
    plt.show()

    print(np.mean(betas_hc))
    print(np.mean(betas_fes))


def betas_distributions_compare(betas_A, betas_B, c_numbers, path, no_hc_A, no_hc_B):

    titles = ['HC-IKEM', 'HC-NUDZ', 'FES-IKEM', 'FES-NUDZ']
    for i in range(len(c_numbers)):
        hc_A = betas_A[0:no_hc_A, i]
        fes_A = betas_A[no_hc_A + 1:, i]
        hc_B = betas_B[0:no_hc_B, i]
        fes_B = betas_B[no_hc_B + 1:, i]

        datasets = [hc_A, hc_B, fes_A, fes_B]

        min_A = min(np.min(hc_A), np.min(fes_A))
        min_B = min(np.min(hc_B), np.min(fes_B))
        max_A = max(np.max(hc_A), np.max(fes_A))
        max_B = max(np.max(hc_B), np.max(fes_B))
        min_edge = min(min_A, min_B)
        max_edge = max(max_A, max_B)

        bins = np.linspace(min_edge, max_edge, 30)

        fig, axs = plt.subplots(2, 2)

        for ax, data, title in zip(axs.ravel(), datasets, titles):
            counts, bins, patches = ax.hist(data, bins=bins, color='darkblue', rwidth=0.85, density=True)
            mean_val = np.mean(data)

            ax.axvline(mean_val, color='red', linestyle='dashed')
            ax.set_title(title)
            ax.set_xlabel('Beta values')
            ax.set_ylabel('Number of subjects')
            ax.set_ylim(0, 1)

        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        fig.suptitle(f'Distribution of beta values, C: {c_numbers[i]}')
        plt.savefig(os.path.join(path, f'betas_distribution_C{c_numbers[i]}'))
        plt.show()


def plot_pdf_cdf(betas_A, betas_B, no_hc_A, no_hc_B, c_numbers, path):

    titles = ["HC-IKEM", "HC-NUDZ", "FES-IKEM", "FES-NUDZ"]
    plt.figure()
    for i in range(len(c_numbers)):
        hc_A = betas_A[0:no_hc_A, i]
        fes_A = betas_A[no_hc_A + 1:, i]
        hc_B = betas_B[0:no_hc_B, i]
        fes_B = betas_B[no_hc_B + 1:, i]

        datasets = [hc_A, hc_B, fes_A, fes_B]
        x = 0
        for data in zip(datasets):
            counts, bins = np.histogram(data, bins=30)
            pdf = counts / np.sum(counts)
            cdf = np.cumsum(pdf)

            plt.plot(cdf, label=titles[x])
            x = x+1

        plt.legend()
        plt.savefig(os.path.join(path, f'cdf_betas_C{c_numbers[i]}'))
        plt.title(f'CDF of beta values for C:  {c_numbers[i]}')
        plt.savefig(os.path.join(path, f'cdf_betas_C{c_numbers[i]}'))
        plt.show()


    return pdf, cdf

