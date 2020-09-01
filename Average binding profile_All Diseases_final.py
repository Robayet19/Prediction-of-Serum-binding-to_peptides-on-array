# The program was written by Robayet. It gives a figure of mean binding distribution of serum Antibody bindings to peptides in different disease cases.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from scipy.stats import kurtosis, skew

median_norm = False  # turn it True if you want to plot median normalized data - default:False
log_scale = True

start = time.time()
if median_norm:
    # data with median normalization
    filename1 = 'sequence_data_NIBIB_Dengue_ML_CTSSeraCare_mod_CV317-Jul-2020-00-08.csv'
    #filename1 = 'sequence_data_NIBIB_Dengue_ML_mod_CV315-Jul-2020-23-54.csv'
    filename2 = 'sequence_data_NIBIB_Normal_ML_mod_CV315-Jul-2020-23-50.csv'
    filename3 = 'sequence_data_NIBIB_WNV_ML_mod_CV315-Jul-2020-23-57.csv'
    filename4 = 'sequence_data_NIBIB_HCV_ML_mod_CV317-Jul-2020-16-50.csv'
    filename5 = 'sequence_data_NIBIB_HBV_ML_mod_CV316-Jul-2020-00-01.csv'
    filename6 = 'sequence_data_NIBIB_Chagas_ML_mod_CV316-Jul-2020-00-02.csv'
    # # data with median normalization
    # filename1 = 'sequence_data_NIBIB_DENV_corr8_ML_mod_18-Nov-2018-19-59.csv'
    # filename2 = 'sequence_data_NIBIB_HBV_corr8_ML_mod_08-Nov-2018-23-05.csv'
    # filename3 = 'sequence_data_NIBIB_HCV_corr8_ML_mod_08-Nov-2018-23-07.csv'
    # filename4 = 'sequence_data_NIBIB_WNV_corr8_ML_mod_20-May-2019-18-12.csv'
    # filename5 = 'sequence_data_NIBIB_Chagas_corr8_ML_mod_08-Nov-2018-17-32.csv'
    # filename6 = 'sequence_data_NIBIB_Normal_corr8_ML_mod_05-Jul-2019-17-33.csv'
else:
    # data without median normalization
    filename1 = 'sequence_data_NIBIB_Dengue_ML_CTSSeraCare_noMed_CV317-Jul-2020-00-10.csv'
    #filename1 = 'sequence_data_NIBIB_Dengue_ML_noMed_CV315-Jul-2020-23-11.csv'
    filename2 = 'sequence_data_NIBIB_Normal_ML_noMed_CV315-Jul-2020-23-35.csv'
    filename3 = 'sequence_data_NIBIB_WNV_ML_noMed_CV315-Jul-2020-23-16.csv'
    filename4 = 'sequence_data_NIBIB_HCV_ML_noMed_CV3_17-Jul-2020-16-56.csv'
    filename5 = 'sequence_data_NIBIB_HBV_ML_noMed_CV315-Jul-2020-23-25.csv'
    filename6 = 'sequence_data_NIBIB_Chagas_ML_noMed_CV315-Jul-2020-23-28.csv'
    # filename1 = 'sequence_data_NIBIB_Dengue_ML_noMed_29-Jun-2020-22-25.csv'
    # filename2 = 'sequence_data_NIBIB_Normal_ML_noMed_29-Jun-2020-22-21.csv'
    # filename3 = 'sequence_data_NIBIB_WNV_ML_noMed_29-Jun-2020-22-27.csv'
    # filename4 = 'sequence_data_NIBIB_HCV_ML_noMed_29-Jun-2020-22-28.csv'
    # filename5 = 'sequence_data_NIBIB_HBV_ML_noMed_29-Jun-2020-22-30.csv'
    # filename6 = 'sequence_data_NIBIB_Chagas_ML_noMed_29-Jun-2020-22-32.csv'

# Import csv (first column is the sequences followed by the binding data)

data1 = pd.read_csv(filename1, header=None)
data2 = pd.read_csv(filename2, header=None)
data3 = pd.read_csv(filename3, header=None)
data4 = pd.read_csv(filename4, header=None)
data5 = pd.read_csv(filename5, header=None)
data6 = pd.read_csv(filename6, header=None)
# take average binding against all peptides for each sample
data1 = data1.iloc[:, 1:]
data1_mean = np.mean(data1, axis=1)
data2 = data2.iloc[:, 1:]
data2_mean = np.mean(data2, axis=1)
data3 = data3.iloc[:, 1:]
data3_mean = np.mean(data3, axis=1)
data4 = data4.iloc[:, 1:]
data4_mean = np.mean(data4, axis=1)
data5 = data5.iloc[:, 1:]
data5_mean = np.mean(data5, axis=1)
data6 = data6.iloc[:, 1:]
data6_mean = np.mean(data6, axis=1)

# combine the average bindings from all states together to make a scatter plot
data_mean = np.transpose(np.vstack((data1_mean, data2_mean, data3_mean, data4_mean, data5_mean, data6_mean)))
state = ['DENV4', 'Control', 'WNV', 'HCV', 'HBV', 'Chagas']
# colors = [colormap(1. * i / len(samples_in_state)) for i in range(len(samples_in_state))]
colors = ['b', 'r', 'y', 'c', 'k', 'g']
color_mean = 'purple'
# plot the binding distribution of average binding from all the states
labels = []
fig1 = plt.figure()
for i in range(data_mean.shape[1]):
    if log_scale:
        # sns.distplot(np.log10(data_mean[:, i]), hist=True, norm_hist=True, color=colors[i], kde=False)
        sns.distplot(np.log10(data_mean[:, i]), norm_hist=True, color=colors[i], hist=True)
        plt.xlabel('log10(Binding intensity)', fontsize=15)
    else:
        # sns.distplot(data_mean[:, i], hist=True, norm_hist=True, color=colors[i], kde=False)
        sns.distplot(data_mean[:, i], norm_hist=True, color=colors[i])
        plt.xlabel('Binding intensity', fontsize=15)

    plt.ylabel('Normalized density', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    labels.append(state[i])
    plt.legend(labels)
    # plt.legend(labels, ncol=1, loc=6, fontsize=12, bbox_to_anchor=[1, 0.5], columnspacing=1.0, labelspacing=1,
    #             handletextpad=0.0, handlelength=2, fancybox=False, shadow=False, borderaxespad=0., mode='expand')

#
# check the kurtosis of each sample's binding distribution
data_tem = data1
num_samples_data = data_tem.shape[1]
kurt_data1 = np.zeros(num_samples_data)
skew_data1 = np.zeros(num_samples_data)
for i in range(num_samples_data):
    kurt_data1[i] = kurtosis(data_tem.iloc[:, i])
    skew_data1[i] = kurtosis(data_tem.iloc[:, i])

# visualize binding distribution of each serum samples in a cohort
# colormap = plt.get_cmap('gist_rainbow')
# colors = [colormap(1. * i / num_samples_data) for i in range(num_samples_data)]
# plt.gca().set_prop_cycle('color', colors)
label_sample = []
fig2 = plt.figure()
for i in range(0, data_tem.shape[1]):
    if log_scale:
        sns.distplot(np.log10(data_tem.iloc[:, i]), norm_hist=True, color=colors[0], hist=False, kde_kws=dict(linewidth=0.2))
        plt.xlabel('log10(Binding intensity)', fontsize=15)
    else:
        sns.distplot(data_tem.iloc[:, i], norm_hist=True, color=colors[0], kde_kws=dict(linewidth=0.2))
        plt.xlabel('Binding intensity', fontsize=15)
    # plot mean distribution
    sns.distplot(np.log10(np.mean(data_tem, axis=1)), norm_hist=False, color=color_mean, hist=False)
    plt.ylabel('Normalized density', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.title('Binding distribution of samples in Dengue4')
    # label_sample.append('S %i'% (i+1))
    # plt.legend(label_sample, loc='best')
    # plt.legend(label_sample, ncol=5, loc='upper center', bbox_to_anchor=[0.5, -0.02], columnspacing=1.0, labelspacing=1,
    #            handletextpad=0.0, handlelength=2, fancybox=False, shadow=False, borderaxespad=0., mode='expand')


data_tem = data2
num_samples_data = data_tem.shape[1]
# check the kurtosis of each sample's binding distribution
kurt_data2 = np.zeros(num_samples_data)
skew_data2 = np.zeros(num_samples_data)
for i in range(num_samples_data):
    kurt_data2[i] = kurtosis(data_tem.iloc[:, i])
    skew_data2[i] = kurtosis(data_tem.iloc[:, i])

# visualize binding distribution of each serum samples in a cohort
# colormap = plt.get_cmap('gist_rainbow')
# colors = [colormap(1. * i / num_samples_data) for i in range(num_samples_data)]
# plt.gca().set_prop_cycle('color', colors)
label_sample = []
fig3 = plt.figure()
for i in range(0, data_tem.shape[1]):
    if log_scale:
        sns.distplot(np.log10(data_tem.iloc[:, i]), norm_hist=True, color=colors[1], hist=False,kde_kws=dict(linewidth=0.2))
        plt.xlabel('log10(Binding intensity)', fontsize=15)
    else:
        sns.distplot(data_tem.iloc[:, i], norm_hist=True, color=colors[1], kde_kws=dict(linewidth=0.2))
        plt.xlabel('Binding intensity', fontsize=15)
    # plot mean distribution
    sns.distplot(np.log10(np.mean(data_tem, axis=1)), norm_hist=False, color=color_mean, hist=False)
    plt.ylabel('Normalized density', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.title('Binding distribution of samples in Control')
    # label_sample.append('S %i'% (i+1))
    # plt.legend(label_sample)
    # plt.legend(label_sample, ncol=5, loc='upper center', bbox_to_anchor=[0.5, -0.05], columnspacing=1.0, labelspacing=1,
    #            handletextpad=0.0, handlelength=2, fancybox=True, shadow=True, borderaxespad=0., mode='expand')


data_tem = data3
num_samples_data = data_tem.shape[1]
# check the kurtosis of each sample's binding distribution
kurt_data3 = np.zeros(num_samples_data)
skew_data3 = np.zeros(num_samples_data)

for i in range(num_samples_data):
    kurt_data3[i] = kurtosis(data_tem.iloc[:, i])
    skew_data3[i] = kurtosis(data_tem.iloc[:, i])
# visualize binding distribution of each serum samples in a cohort
# colormap = plt.get_cmap('gist_rainbow')
# colors = [colormap(1. * i / num_samples_data) for i in range(num_samples_data)]
# plt.gca().set_prop_cycle('color', colors)
label_sample = []
fig4 = plt.figure()
for i in range(0, data_tem.shape[1]):
    if log_scale:
        sns.distplot(np.log10(data_tem.iloc[:, i]), norm_hist=True, color=colors[2], hist=False, kde_kws=dict(linewidth=0.2))
        plt.xlabel('log10(Binding intensity)', fontsize=15)
    else:
        sns.distplot(data_tem.iloc[:, i], norm_hist=True, color=colors[2], kde_kws=dict(linewidth=0.2))
        plt.xlabel('Binding intensity', fontsize=15)
    # plot mean distribution
    sns.distplot(np.log10(np.mean(data_tem, axis=1)), norm_hist=False, color=color_mean, hist=False)
    plt.ylabel('Normalized density', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.title('Binding distribution of samples in WNV')
    # label_sample.append('S %i'% (i+1))
    # plt.legend(label_sample)
    # plt.legend(label_sample, ncol=5, loc='upper center', bbox_to_anchor=[0.5, -0.05], columnspacing=1.0, labelspacing=1,
    #            handletextpad=0.0, handlelength=2, fancybox=True, shadow=True, borderaxespad=0., mode='expand')


data_tem = data4
num_samples_data = data_tem.shape[1]
# check the kurtosis of each sample's binding distribution
kurt_data4 = np.zeros(num_samples_data)
skew_data4 = np.zeros(num_samples_data)
for i in range(num_samples_data):
    kurt_data4[i] = kurtosis(data_tem.iloc[:, i])
    skew_data4[i] = kurtosis(data_tem.iloc[:, i])

# visualize binding distribution of each serum samples in a cohort
# colormap = plt.get_cmap('gist_rainbow')
# colors = [colormap(1. * i / num_samples_data) for i in range(num_samples_data)]
# plt.gca().set_prop_cycle('color', colors)
label_sample = []
fig5 = plt.figure()
for i in range(0, data_tem.shape[1]):
    if log_scale:
        sns.distplot(np.log10(data_tem.iloc[:, i]), norm_hist=True, color=colors[3], hist=False, kde_kws=dict(linewidth=0.2))
        plt.xlabel('log10(Binding intensity)', fontsize=15)
    else:
        sns.distplot(data_tem.iloc[:, i], norm_hist=True, color=colors[3], kde_kws=dict(linewidth=0.2))
        plt.xlabel('Binding intensity', fontsize=15)
    # plot mean distribution
    sns.distplot(np.log10(np.mean(data_tem, axis=1)), norm_hist=False, color=color_mean, hist=False)
    plt.ylabel('Normalized density', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.title('Binding distribution of samples in HCV')
    # label_sample.append('S %i'% (i+1))
    # plt.legend(label_sample)
    # plt.legend(label_sample, ncol=5, loc='upper center', bbox_to_anchor=[0.5, -0.05], columnspacing=1.0, labelspacing=1,
    #            handletextpad=0.0, handlelength=2, fancybox=True, shadow=True, borderaxespad=0., mode='expand')

data_tem = data5
num_samples_data = data_tem.shape[1]
# check the kurtosis of each sample's binding distribution
kurt_data5 = np.zeros(num_samples_data)
skew_data5 = np.zeros(num_samples_data)
for i in range(num_samples_data):
    kurt_data5[i] = kurtosis(data_tem.iloc[:, i])
    skew_data5[i] = kurtosis(data_tem.iloc[:, i])

# colormap = plt.get_cmap('gist_rainbow')
# colors = [colormap(1. * i / num_samples_data) for i in range(num_samples_data)]
# plt.gca().set_prop_cycle('color', colors)
label_sample = []
fig6 = plt.figure()
for i in range(0, data_tem.shape[1]):
    if log_scale:
        sns.distplot(np.log10(data_tem.iloc[:, i]), norm_hist=True, color=colors[4], hist=False, kde_kws=dict(linewidth=0.2))
        plt.xlabel('log10(Binding intensity)', fontsize=15)
    else:
        sns.distplot(data_tem.iloc[:, i], norm_hist=True, color=colors[4], kde_kws=dict(linewidth=0.2))
        plt.xlabel('Binding intensity', fontsize=15)
    # plot mean distribution
    sns.distplot(np.log10(np.mean(data_tem, axis=1)), norm_hist=False, color=color_mean, hist=False)
    plt.ylabel('Normalized density', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.title('Binding distribution of samples in HBV')

data_tem = data6
num_samples_data = data_tem.shape[1]
# check the kurtosis of each sample's binding distribution
kurt_data6 = np.zeros(num_samples_data)
skew_data6 = np.zeros(num_samples_data)
for i in range(num_samples_data):
    kurt_data6[i] = kurtosis(data_tem.iloc[:, i])
    skew_data6[i] = kurtosis(data_tem.iloc[:, i])

# colormap = plt.get_cmap('gist_rainbow')
# colors = [colormap(1. * i / num_samples_data) for i in range(num_samples_data)]
# plt.gca().set_prop_cycle('color', colors)
label_sample = []
fig7 = plt.figure()
for i in range(0, data_tem.shape[1]):
    if log_scale:
        sns.distplot(np.log10(data_tem.iloc[:, i]), norm_hist=True, color=colors[5], hist=False, kde_kws=dict(linewidth=0.2))
        plt.xlabel('log10(Binding intensity)', fontsize=15)
    else:
        sns.distplot(data_tem.iloc[:, i], norm_hist=True, color=colors[5], kde_kws=dict(linewidth=0.2))
        plt.xlabel('Binding intensity', fontsize=15)
    # plot mean distribution
    sns.distplot(np.log10(np.mean(data_tem, axis=1)), norm_hist=False, color=color_mean, hist=False)
    plt.ylabel('Normalized density', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.title('Binding distribution of samples in Chagas')

end = time.time()
print(f'\nTime to complete:{end-start:.2f}s\n')
plt.show()
