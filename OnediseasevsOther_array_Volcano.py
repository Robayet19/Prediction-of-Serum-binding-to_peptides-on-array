# This program was written Robayet Chowdhury

from typing import Union, Iterable

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import multiprocessing
from multiprocessing import Pool
#from numpy.core._multiarray_umath import ndarray
from scipy.stats import gaussian_kde
import scipy.stats as stats

median_norm = True
logscale = True
Z_scale = False
cutoff_value1 = -0.2  # asssign a value to show the first cut-off value on x axis
cutoff_value2 = 0.2  # asssign a value to show the second cut-off value on x axis
alpha = 0.05  # This will be used to calculate bondferroni cut-off value
case_name = ['DENV4', 'Control', 'WNV', 'HCV', 'HBV', 'Chagas']
contrast_name = [case_name[0] + ' - ' + case_name[1], case_name[0] + ' - ' + case_name[2],
                 case_name[0] + ' - ' + case_name[3], case_name[0] + ' - ' + case_name[4],
                 case_name[0] + ' - ' + case_name[5], case_name[2] + ' - ' + case_name[1],
                 case_name[3] + ' - ' + case_name[1], case_name[4] + ' - ' + case_name[1],
                 case_name[5] + ' - ' + case_name[1], case_name[2] + ' - ' + case_name[3],
                 case_name[2] + ' - ' + case_name[4], case_name[2] + ' - ' + case_name[5],
                 case_name[3] + ' - ' + case_name[4], case_name[3] + ' - ' + case_name[5],
                 case_name[4] + ' - ' + case_name[5]]

if median_norm:
    # data with median normalization
    filename1 = 'sequence_data_NIBIB_Dengue_ML_CTSSeraCare_mod_CV317-Jul-2020-00-08.csv'
    filename1 = 'sequence_data_NIBIB_Dengue_ML_mod_CV315-Jul-2020-23-54.csv'
    filename2 = 'sequence_data_NIBIB_Normal_ML_mod_CV315-Jul-2020-23-50.csv'
    filename3 = 'sequence_data_NIBIB_WNV_ML_mod_CV315-Jul-2020-23-57.csv'
    filename4 = 'sequence_data_NIBIB_HCV_ML_mod_CV317-Jul-2020-16-50.csv'
    filename5 = 'sequence_data_NIBIB_HBV_ML_mod_CV316-Jul-2020-00-01.csv'
    filename6 = 'sequence_data_NIBIB_Chagas_ML_mod_CV316-Jul-2020-00-02.csv'

    # filename1 = 'sequence_data_NIBIB_Dengue_ML_mod_11-Apr-2020-13-59.csv'
    # filename2 = 'sequence_data_NIBIB_Normal_ML_mod_21-Jun-2020-00-59.csv'
    # filename3 = 'sequence_data_NIBIB_WNV_ML_mod_21-Jun-2020-00-48.csv'
    # filename4 = 'sequence_data_NIBIB_HCV_ML_mod_21-Jun-2020-00-56.csv'
    # filename5 = 'sequence_data_NIBIB_HBV_ML_mod_21-Jun-2020-00-57.csv'
    # filename6 = 'sequence_data_NIBIB_Chagas_ML_mod_21-Jun-2020-00-46.csv'

else:
    # data without median normalization
    #filename1 = 'sequence_data_NIBIB_Dengue_ML_noMed_CV315-Jul-2020-23-11.csv'
    filename1 = 'sequence_data_NIBIB_Dengue_ML_CTSSeraCare_noMed_CV317-Jul-2020-00-10.csv'
    filename2 = 'sequence_data_NIBIB_Normal_ML_noMed_CV315-Jul-2020-23-35.csv'
    filename3 = 'sequence_data_NIBIB_WNV_ML_noMed_CV315-Jul-2020-23-16.csv'
    filename4 = 'sequence_data_NIBIB_HCV_ML_noMed_CV3_17-Jul-2020-16-56.csv'
    filename5 = 'sequence_data_NIBIB_HBV_ML_noMed_CV315-Jul-2020-23-25.csv'
    filename6 = 'sequence_data_NIBIB_Chagas_ML_noMed_CV315-Jul-2020-23-28.csv'
    #
    # filename1 = 'sequence_data_NIBIB_Dengue_ML_noMed_29-Jun-2020-22-25.csv'
    # filename2 = 'sequence_data_NIBIB_Normal_ML_noMed_29-Jun-2020-22-21.csv'
    # filename3 = 'sequence_data_NIBIB_WNV_ML_noMed_29-Jun-2020-22-27.csv'
    # filename4 = 'sequence_data_NIBIB_HCV_ML_noMed_29-Jun-2020-22-28.csv'
    # filename5 = 'sequence_data_NIBIB_HBV_ML_noMed_29-Jun-2020-22-30.csv'
    # filename6 = 'sequence_data_NIBIB_Chagas_ML_noMed_29-Jun-2020-22-32.csv'

# read all files
data1 = pd.read_csv(filename1, header=None)
data2 = pd.read_csv(filename2, header=None)
data3 = pd.read_csv(filename3, header=None)
data4 = pd.read_csv(filename4, header=None)
data5 = pd.read_csv(filename5, header=None)
data6 = pd.read_csv(filename6, header=None)

total_peptides = data1.shape[0]
bf_cutoff = alpha / total_peptides  # calculate the bonferroni correction
# separate binding data and convert into log scale
if logscale:
    data1 = np.log10(data1.iloc[:, 1:])
    data2 = np.log10(data2.iloc[:, 1:])
    data3 = np.log10(data3.iloc[:, 1:])
    data4 = np.log10(data4.iloc[:, 1:])
    data5 = np.log10(data5.iloc[:, 1:])
    data6 = np.log10(data6.iloc[:, 1:])
else:
    data1 = data1.iloc[:, 1:]
    data2 = data2.iloc[:, 1:]
    data3 = data3.iloc[:, 1:]
    data4 = data4.iloc[:, 1:]
    data5 = data5.iloc[:, 1:]
    data6 = data6.iloc[:, 1:]
# take the average and calculate the difference between different contrasts
data_1vs2 = np.mean(data1, axis=1) - np.mean(data2, axis=1)
data_1vs3 = np.mean(data1, axis=1) - np.mean(data3, axis=1)
data_1vs4 = np.mean(data1, axis=1) - np.mean(data4, axis=1)
data_1vs5 = np.mean(data1, axis=1) - np.mean(data5, axis=1)
data_1vs6 = np.mean(data1, axis=1) - np.mean(data6, axis=1)
data_3vs2 = np.mean(data3, axis=1) - np.mean(data2, axis=1)
data_4vs2 = np.mean(data4, axis=1) - np.mean(data2, axis=1)
data_5vs2 = np.mean(data5, axis=1) - np.mean(data2, axis=1)
data_6vs2 = np.mean(data6, axis=1) - np.mean(data2, axis=1)
data_3vs4 = np.mean(data3, axis=1) - np.mean(data4, axis=1)
data_3vs5 = np.mean(data3, axis=1) - np.mean(data5, axis=1)
data_3vs6 = np.mean(data3, axis=1) - np.mean(data6, axis=1)
data_4vs5 = np.mean(data4, axis=1) - np.mean(data5, axis=1)
data_4vs6 = np.mean(data4, axis=1) - np.mean(data6, axis=1)
data_5vs6 = np.mean(data5, axis=1) - np.mean(data6, axis=1)

# calculate the ratio between different contrasts
# data_1vs2 = np.mean(data1, axis=1) / np.mean(data2, axis=1)
# data_1vs3 = np.mean(data1, axis=1) / np.mean(data3, axis=1)
# data_1vs4 = np.mean(data1, axis=1) / np.mean(data4, axis=1)
# data_1vs5 = np.mean(data1, axis=1) / np.mean(data5, axis=1)
# data_1vs6 = np.mean(data1, axis=1) / np.mean(data6, axis=1)
# data_3vs2 = np.mean(data3, axis=1) / np.mean(data2, axis=1)
# data_4vs2 = np.mean(data4, axis=1) / np.mean(data2, axis=1)
# data_5vs2 = np.mean(data5, axis=1) / np.mean(data2, axis=1)
# data_6vs2 = np.mean(data6, axis=1) / np.mean(data2, axis=1)
# data_3vs4 = np.mean(data3, axis=1) / np.mean(data4, axis=1)
# data_3vs5 = np.mean(data3, axis=1) / np.mean(data5, axis=1)
# data_3vs6 = np.mean(data3, axis=1) / np.mean(data6, axis=1)
# data_4vs5 = np.mean(data4, axis=1) / np.mean(data5, axis=1)
# data_4vs6 = np.mean(data4, axis=1) / np.mean(data6, axis=1)
# data_5vs6 = np.mean(data5, axis=1) / np.mean(data6, axis=1)

# combine all values of all contrasts into one
mean_contrast_all = ([data_1vs2], [data_1vs3], [data_1vs4], [data_1vs5], [data_1vs6], [data_3vs2], [data_4vs2],
                 [data_5vs2], [data_6vs2], [data_3vs4], [data_3vs5], [data_3vs6], [data_4vs5], [data_4vs6], [data_5vs6])


def Z_score_calc(X, Y):
    std_1 = np.std(X, axis=1)
    std_2 = np.std(Y, axis=1)
    mean_1 = np.mean(X, axis=1)
    mean_2 = np.mean(Y, axis=1)
    n1 = X.shape[1]
    n2 = Y.shape[1]
    Z_score = (mean_1 - mean_2) / np.sqrt((std_1 ** 2 / n1) + (std_2 ** 2 / n2))
    # Z_score = (mean_1 - mean_2) / np.sqrt(std_1**2 + std_2**2)
    return Z_score


if Z_scale:
    # calculate Z score between two cohorts
    Z_1v2 = Z_score_calc(data1, data2)
    Z_1v3 = Z_score_calc(data1, data3)
    Z_1v4 = Z_score_calc(data1, data4)
    Z_1v5 = Z_score_calc(data1, data5)
    Z_1v6 = Z_score_calc(data1, data6)
    Z_3v2 = Z_score_calc(data3, data2)
    Z_4v2 = Z_score_calc(data4, data2)
    Z_5v2 = Z_score_calc(data5, data2)
    Z_6v2 = Z_score_calc(data6, data2)
    Z_3v4 = Z_score_calc(data3, data4)
    Z_3v5 = Z_score_calc(data3, data5)
    Z_3v6 = Z_score_calc(data3, data6)
    Z_4v5 = Z_score_calc(data4, data5)
    Z_4v6 = Z_score_calc(data4, data6)
    Z_5v6 = Z_score_calc(data5, data6)
    # combine Z scores from different contrasts together
    file_run = ([Z_1v2], [Z_1v3], [Z_1v4], [Z_1v5], [Z_1v6], [Z_3v2], [Z_4v2], [Z_5v2], [Z_6v2], [Z_3v4], [Z_3v5],
                [Z_3v6], [Z_4v5], [Z_4v6], [Z_5v6])
else:
    # calculate the p-value
    _, pval_1v2 = stats.ttest_ind(data1, data2, axis=1, equal_var=False)
    _, pval_1v3 = stats.ttest_ind(data1, data3, axis=1, equal_var=False)
    _, pval_1v4 = stats.ttest_ind(data1, data4, axis=1, equal_var=False)
    _, pval_1v5 = stats.ttest_ind(data1, data5, axis=1, equal_var=False)
    _, pval_1v6 = stats.ttest_ind(data1, data6, axis=1, equal_var=False)
    _, pval_3v2 = stats.ttest_ind(data3, data2, axis=1, equal_var=False)
    _, pval_4v2 = stats.ttest_ind(data4, data2, axis=1, equal_var=False)
    _, pval_5v2 = stats.ttest_ind(data5, data2, axis=1, equal_var=False)
    _, pval_6v2 = stats.ttest_ind(data6, data2, axis=1, equal_var=False)
    _, pval_3v4 = stats.ttest_ind(data3, data4, axis=1, equal_var=False)
    _, pval_3v5 = stats.ttest_ind(data3, data5, axis=1, equal_var=False)
    _, pval_3v6 = stats.ttest_ind(data3, data6, axis=1, equal_var=False)
    _, pval_4v5 = stats.ttest_ind(data4, data5, axis=1, equal_var=False)
    _, pval_4v6 = stats.ttest_ind(data4, data6, axis=1, equal_var=False)
    _, pval_5v6 = stats.ttest_ind(data5, data6, axis=1, equal_var=False)
    # combine the p-values of all contrasts together
    file_run_save = ([pval_1v2], [pval_1v3], [pval_1v4], [pval_1v5], [pval_1v6], [pval_3v2], [pval_4v2], [pval_5v2],
                         [pval_6v2], [pval_3v4], [pval_3v5], [pval_3v6], [pval_4v5], [pval_4v6], [pval_5v6])
    file_run = np.log10(file_run_save)


# This function counts the no of significant and high binding peptides above cutoff value in each cohort


def significance(AverageDifference, pvalues, cutoff1, cutoff2, bf_cutoff, total_peptides):
    signif_Numpep1 = 0
    signif_Numpep2 = 0
    for i in range(0, total_peptides):
        if AverageDifference[i] > cutoff2 and pvalues[i] <= bf_cutoff:
            signif_Numpep1 = signif_Numpep1 + 1
        if AverageDifference[i] <= cutoff1 and pvalues[i] <= bf_cutoff:
        #if AverageRatio[i] <= cutoff1 and pvalues[i] <= bf_cutoff:
            signif_Numpep2 = signif_Numpep2 + 1
    return signif_Numpep1, signif_Numpep2


# count the no of statistically significant peptides with mean binding values above the two cut-off values
if not Z_scale:
    num_pep_contrast = np.zeros((len(mean_contrast_all), 2))
    for i in range(len(mean_contrast_all)):
        numPep_cohorts = significance(mean_contrast_all[i][0], file_run_save[i][0], cutoff_value1, cutoff_value2, bf_cutoff,
                                      total_peptides)
        num_pep_contrast[i, :] = numPep_cohorts

# calculate density
density_array = np.zeros((total_peptides, len(file_run)))


def density_calculation(data_set):
    print('running data set:', data_set + 1)
    # data = file_run[data_set]
    data = mean_contrast_all[data_set][0]  # use difference in average binding or average ratio to calculate density
    density = gaussian_kde(data)(data)
    return density


if __name__ == '__main__':
    start = time.time()
    pool = Pool()
    # result = pool.map(density_calculation, range(len(file_run)))
    result = pool.map(density_calculation, range(len(mean_contrast_all)))
    for i, j in enumerate(result):
        density_array[:, i] = j
    del result
    end = time.time()
    print(f'\nTime to complete:{end - start:.2f}s\n')

# Scatter Plots colored by the density
for i in range(len(mean_contrast_all)):
    fig, ax = plt.subplots()
    if Z_scale:
        ax.scatter(mean_contrast_all[i][0], file_run[i], c=density_array[:, i], s=5, edgecolor='')
        # plt.ylabel('Z-scores', fontsize=12)
    else:
        ax.scatter(mean_contrast_all[i][0], -file_run[i], c=density_array[:, i], s=5, edgecolor='')
        # draw a horizontal line to show the Bonferroni cut-off value  on Y axis
        plt.axhline(-np.log10(bf_cutoff), color='r', linestyle='--')
        # plt.ylabel('-log10(P-value', fontsize=12)
    # plt.xlabel('Difference  average bindings', fontsize=12)
    plt.xticks(np.arange(-1, 1 + 0.2, 0.2))
    plt.axvline(cutoff_value1, color='k', linestyle='--')  # draw first vertical line to show the cut-off value on x axis
    plt.axvline(cutoff_value2, color='k', linestyle='--')  # draw second  vertical line to show the cut-off value on x axis
    # generate customized legend and title
    handles, labels = plt.gca().get_legend_handles_labels()  # get existing handles and labels
    contrast = contrast_name[i]
    empty_patch = mpatches.Patch(color='None', label=contrast)  # create a patch with no color
    handles.append(empty_patch)  # add new patches and labels to list
    labels.append(contrast)
    plt.legend(handles, labels, loc=1, frameon=True)
    contrast = contrast.replace('-', 'Vs.')  # replace '-'  with 'Vs.' for the title
    plt.title(contrast, fontsize=14)

# plt.show()
