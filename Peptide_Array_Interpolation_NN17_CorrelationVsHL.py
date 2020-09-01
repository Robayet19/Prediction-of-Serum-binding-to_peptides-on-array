# This program was originally written Dr. Alexandar Taguchi and later significantly modified by Robayet chowdhury
# The program gives a set of optimized parameters such as combination of hidden layers and hidden nodes and evaluate the performance using Pearson Correlation Coefficient (R)

# ---Neural Network for Peptide Array Interpolation--- #
#
# Architecture:
#    Input: Sequence (residue number x amino acid type matrix),
#    Neural Network: 1 fully connected hidden layer to encode
#        amino acid eigenvectors (w/o relu or bias), a flattening
#        operation, multiple fully connected hidden layers (w/o bias),
#        and 1 regression output layer
#    Output: Predicted binding value (float)
#
# Training: The neural network is trained in PyTorch on a subset
#    of the sequences where the binding data is not saturated.
#    The model is then tested on the remaining sequences that
#    include the saturated data points.
#
# Options:
#    - Automatic removal of GSG from the end of peptides
#    - Removal of short and/or empty sequences
#    - Fitting in log10 or linear space
#    - Uniform sampling of data range during training
#    - Train all in parallel, individuals, or average all samples
#    - GPU support


# ~~~~~~~~~~MODULES~~~~~~~~~~ #

import argparse
import concurrent.futures
import datetime
import multiprocessing
from multiprocessing import Pool
from scipy.stats import gaussian_kde
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from tkinter import * # this is required to input a file from a directory using GUI
from tkinter.filedialog import askopenfilename
import time
t = time.time()

# Create dictionary of imported parameters
parser = argparse.ArgumentParser()
parser.add_argument('params', nargs='?')  # nargs='?' means 0-or-1 arguments.' add_argument' method adds parameter to
# the module.
args = parser.parse_args()

# Create dictionary of imported parameters
paramImport = {}
if args.params:
    with open(args.params, 'r') as paramFile:
        for row in paramFile.readlines():
            key, value = row.split(',')
            value = value[:-1]
            if value.isdigit():
                param = int(value)
            elif value[0].isdigit():
                param = float(value)
            elif value.lower() == 'true':
                param = True
            elif value.lower() == 'false':
                param = False
            else:
                param = value
            paramImport[key] = param

# Store current module and parameter scope
moduleScope = dir()

# ~~~~~~~~~~MAIN PARAMETERS~~~~~~~~~~ #
Bias_inputLayer = True  # add bias to amino layer  -default: False
Bias_HiddenLayer = True  # add bias to hiddenlayers -default: False
aminoEigen = 5  # number of features to describe amino acids - default: 10
drop_prob = 0.1
#filename = 'sequence_data_NIBIB_Dengue_ML_CTSSeraCare_mod_CV317-Jul-2020-00-08.csv' # import sequence and binding data
# filename = 'sequence_data_NIBIB_Normal_ML_mod_CV315-Jul-2020-23-50.csv'
# filename = 'sequence_data_NIBIB_WNV_ML_mod_CV315-Jul-2020-23-57.csv'
# filename = 'sequence_data_NIBIB_HCV_ML_mod_CV317-Jul-2020-16-50.csv'
# filename = 'sequence_data_NIBIB_HBV_ML_mod_CV316-Jul-2020-00-01.csv'
# filename = 'sequence_data_NIBIB_Chagas_ML_mod_CV316-Jul-2020-00-02.csv'
# filename = 'NIBIB_Dengue4(CTSSera)ND_Zscore_multisample_eval_CV3.csv'
# filename = 'NIBIB_DENV(CTSSera)vsChagas_CV3_Zscore_multisample_eval.csv'
#filename = 'NIBIB_DENV(CTSSera)vsHBV_CV3_Zscore_multisample_eval.csv'
# filename = 'NIBIB_DENV(CTSSera)vsHCV_CV3_Zscore_multisample_eval.csv'
# filename = 'NIBIB_DENV(CTSSera)vsWNV_CV3_Zscore_multisample_eval.csv'
# filename = 'NIBIB_WNVvsND_CV3_Zscore_multisample_eval.csv'
# filename = 'NIBIB_WNVvsHCV_CV3_Zscore_multisample_eval.csv'
# filename = 'NIBIB_WNVvsHBV_CV3_Zscore_multisample_eval.csv'
# filename = 'NIBIB_WNVvsChagas_CV3_Zscore_multisample_eval.csv'
# filename = 'NIBIB_HCVvsChagas_Zscore_CV3_multisample_eval.csv'
# filename = 'NIBIB_HCVvsND_Zscore_CV3_multisample_eval.csv'
# filename = 'NIBIB_HCVvsHBV_Zscore_CV3_multisample_eval.csv'
# filename = 'NIBIB_HBVND_Z score_CV3_multisample_eval.csv'
filename = 'NIBIB_HBVvsChagas_Zscore_CV3_multisample_eval.csv'
# filename = 'NIBIB_ChagasvsND_Zscore_CV3_multisample_eval.csv'
hiddenLayers = 1  # number of hidden layers - default: 1
hiddenWidth = 100  # width of hidden layers - default: 100
trainSteps = 50000  # number of training steps - default: 20000
trainingMode = 1  # train all ('all'), individuals (1, 2, 3...), or average ('avg')
weightFolder = 'NIBIB_HBV_CorrVsHL'  # name of folder for saving weights and biases
weightSave = True  # save weights to file - default: False

# ~~~~~~~~~~ADDITIONAL PARAMETERS~~~~~~~~~~ #
aminoAcids = 'ADEFGHKLNPQRSVWY'  # letter codes - default: 'ADEFGHKLNPQRSVWY'
batch = 100  # batch size for training - default: 100
epoch = 3  # no of times the dataset to be run - default: 2
dataSaturation = 0.995  # saturation threshold for training - default: 0.995
dataShift = False  # added to data before log10 (False for linear) - default: 100
dropOut_regularization = False
gpuTraining = False  # allow for GPU training - default: False
learnRate = 0.001  # magnitude of gradient descent step - default: 0.001
minResidues = 0  # minimum sequence length
stepPrint = 200  # step multiple at which to print progress - default: 100
stepBatch = 100  # batch size for train/test progress report - default: 1000
tolerance = 0.1  # bin offset that still counts as a hit - default: 0.2
trainFinal = 10  # final optimization steps on all training data - default: 10
trainFraction = 0.9  # fraction of non-saturated data for training - default: 0.9
uniformBGD = True  # uniform batch gradient descent - default: True
weightRegularization = 0  # least-squares weight regularization - default: 0
showFigure = False

# Create arrays for hidden layers, hidden widths, and correlations
HL_array = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
HW_array = (10, 50, 100, 150, 200, 250, 300, 350)

paramScope = [x for x in dir() if x not in moduleScope + ['moduleScope']]

# Replace parameters with imported settings
if paramImport:
    globals().update(paramImport)
paramLocals = locals()
paramDict = {x: paramLocals[x] for x in paramScope}
new_file = '/CorrVsHL_HBVvsChagas_Zfit.txt'


if trainingMode == 'avg':
    sample = 1
elif type(trainingMode) == int:
    sample = trainingMode

# create a zero matrix to save the correlation values from each set of parameters
Corr_array = np.zeros((len(HL_array), len(HW_array), epoch))
load_array = False

cc
def training(params):
    hiddenLayers = params[0]
    hiddenWidth = params[1]
    sample = params[2]
    rep_num = params[3]
    replicate_num = rep_num + 1
    print('No of hidden layers: %.f|No of hidden width: %.f|Replicate no: %.f' %
          (hiddenLayers, hiddenWidth, replicate_num))

    # update main parameters
    paramDict.update(hiddenLayers=hiddenLayers)
    paramDict.update(hiddenWidth=hiddenWidth)
    paramDict.update(replicate_num=replicate_num)
    # Import matplotlib into new environment
    import matplotlib.pyplot as plt

    # Import csv (first column is the sequences followed by the binding data)
    data = pd.read_csv(filename, header=None)

    # Extract column for assigning training and testing sets
    data_split = np.array([], dtype=int)
    if set(data.iloc[:, -1]) == {0, 1}:
        data_split = data.iloc[:, -1].values
        data = data.iloc[:, :-1]

    # print('replacing 4 unused amino acids with the ones on array')
    # replace 4 unused amino acids with the ones on array if you input sequences of protein/proteome
    data[0] = data[0].str.replace('I', 'V')
    data[0] = data[0].str.replace('T', 'S')
    data[0] = data[0].str.replace('M', 'L')
    data[0] = data[0].str.replace('C', 'A')

    # Check for bad sequence entries
    data[0].replace(re.compile('[^' + aminoAcids + ']'), '', inplace=True)

    # Remove trailing GSG from sequences
    if sum(data[0].str[-3:] == 'GSG') / len(data) > 0.9:
        data[0] = data[0].str[:-3]

    # Find length of longest string
    max_len = data[0].str.len().max()

    # Remove short sequences
    data.drop(data[0].index[data[0].str.len() < minResidues].tolist(), inplace=True)

    # Assign binary vector to each amino acid
    amino_dict = {n: m for (m, n) in enumerate(aminoAcids)}

    # Create binary sequence matrix representation
    sequences = np.zeros((len(data), len(aminoAcids) * max_len), dtype='int8')
    for (n, m) in enumerate(data[0]):
        amino_ind = [amino_dict[j] + (i * len(aminoAcids)) for (i, j) in enumerate(m)]
        sequences[n][amino_ind] = 1

    total_samples = data.shape[1]

    # Train all, one, or averaged data
    if sample > 0:
        data = data[sample].values
    else:
        data = np.mean(data.loc[:, 1:].values, axis=1)

    # Apply baseline shift and base-10 logarithm , and update dataShift
    if type(dataShift) == float and total_samples >= 1:
        print('adding dataShift')
        data = np.log10(data + dataShift)
        # paramDict.update(dataShift=dataShift)
    elif bool(dataShift == True) and total_samples >= 1:
        print('subtracting min value and adding 1')
        data = np.log10(data - (min(data) - 1))
        # paramDict.update(dataShift=-(min(data) - 1))

    # Concatenate sequence and data
    train_test = np.concatenate((sequences, np.transpose([data])), axis=1)

    # Assign training and testings sets manually
    if len(data_split):

        # Training set
        train_xy = np.copy(train_test[~data_split.astype(bool), :])
        train_xy = train_xy[train_xy[:, -1].argsort()]

        # Test set
        test_xy = np.copy(train_test[data_split.astype(bool), :])

        # Assign training and testings sets randomly
    else:

        # Shuffle all data below saturation threshold
        train_test = train_test[train_test[:, -1].argsort()]
        saturate_ind = np.abs(train_test[:, -1] - dataSaturation * train_test[-1, -1]).argmin()
        np.random.shuffle(train_test[:saturate_ind])

        # Training set
        train_xy = np.copy(train_test[:int(trainFraction * saturate_ind), :])
        train_xy = train_xy[train_xy[:, -1].argsort()]

        # Test set
        test_xy = np.copy(train_test[int(trainFraction * saturate_ind):, :])

    # Find bin indices for uniformly distributed batch gradient descent
    bin_data = np.linspace(train_xy[0][-1], train_xy[-1][-1], batch)
    bin_ind = [np.argmin(np.abs(x - train_xy[:, -1])) for x in bin_data]
    bin_ind = np.append(bin_ind, len(train_xy))

    # GPU Training
    if gpuTraining:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

        # Convert to PyTorch variable tensors
    train_seq = torch.from_numpy(train_xy[:, :-1]).float().to(device)
    train_data = torch.from_numpy(train_xy[:, -1]).float().to(device)
    test_seq = torch.from_numpy(test_xy[:, :-1]).float().to(device)
    test_data = torch.from_numpy(test_xy[:, -1]).float().to(device)

    # ~~~~~~~~~~NEURAL NETWORK~~~~~~~~~~ #

    class NeuralNet(nn.Module):
        if dropOut_regularization:
            def __init__(self, width, layers, drop):
                super().__init__()

                # Network layers
                if Bias_inputLayer:
                    self.AminoLayer = nn.Linear(len(aminoAcids), aminoEigen, bias=True)
                else:
                    self.AminoLayer = nn.Linear(len(aminoAcids), aminoEigen, bias=False)
                if Bias_HiddenLayer:
                    self.HiddenLayers = nn.ModuleList(
                        [nn.Linear(int(max_len * aminoEigen), width, bias=True)])
                    self.HiddenLayers.extend([nn.Linear(width, width, bias=True) for _ in range(layers - 1)])
                else:
                    self.HiddenLayers = nn.ModuleList(
                        [nn.Linear(int(max_len * aminoEigen), width, bias=False)])
                    self.HiddenLayers.extend(
                        [nn.Linear(width, width, bias=False) for _ in range(layers - 1)])

                self.OutputLayer = nn.Linear(width, 1, bias=True)
                self.dropout = nn.Dropout(p=drop)

            def forward(self, seq):
                out = seq.view(-1, len(aminoAcids))
                out = self.AminoLayer(out)
                out = out.view(-1, int(max_len * aminoEigen))
                for x in self.HiddenLayers:
                    out = self.dropout(functional.relu(x(out)))
                out = self.OutputLayer(out)
                return out
        else:
            def __init__(self, width, layers):
                super().__init__()

                # Network layers
                if Bias_inputLayer:
                    self.AminoLayer = nn.Linear(len(aminoAcids), aminoEigen, bias=True)
                else:
                    self.AminoLayer = nn.Linear(len(aminoAcids), aminoEigen, bias=False)
                if Bias_HiddenLayer:
                    self.HiddenLayers = nn.ModuleList([nn.Linear(int(max_len * aminoEigen), width, bias=True)])
                    self.HiddenLayers.extend([nn.Linear(width, width, bias=True) for _ in range(layers - 1)])
                else:
                    self.HiddenLayers = nn.ModuleList([nn.Linear(int(max_len * aminoEigen), width, bias=False)])
                    self.HiddenLayers.extend([nn.Linear(width, width, bias=False) for _ in range(layers - 1)])

                self.OutputLayer = nn.Linear(width, 1, bias=True)

            def forward(self, seq):
                out = seq.view(-1, len(aminoAcids))
                out = self.AminoLayer(out)
                out = out.view(-1, int(max_len * aminoEigen))
                for x in self.HiddenLayers:
                    out = functional.relu(x(out))
                out = self.OutputLayer(out)
                return out

    if dropOut_regularization:
        net = NeuralNet(width=hiddenWidth, layers=hiddenLayers, drop=drop_prob).to(device)
    else:
        net = NeuralNet(width=hiddenWidth, layers=hiddenLayers).to(device)
    print('\nARCHITECTURE:')
    print(net)

    # Loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learnRate, weight_decay=weightRegularization)

    # ~~~~~~~~~~TRAINING~~~~~~~~~~ #
    best_net = net
    best_loss = np.inf
    print('\nTRAINING:')
    train_test_error = np.zeros((int(trainSteps / stepPrint), 2))
    for i in range(trainSteps + 1):

        # Select indices for training
        if uniformBGD:
            train_ind = [np.random.randint(bin_ind[i], bin_ind[i + 1] + 1)
                         for i in range(len(bin_ind) - 1)]
            train_ind[-1] = train_ind[-1] - 1
        else:
            train_ind = random.sample(range(train_data.shape[0]), batch)

        # Calculate loss
        if i < (trainSteps - trainFinal):
            train_out = net(train_seq[train_ind])
            loss = loss_function(torch.squeeze(train_out), train_data[train_ind])
        else:
            train_out = net(train_seq)
            loss = loss_function(torch.squeeze(train_out), train_data)

            # Remember the best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_net = net

        # Weight optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Report training progress
        if i % stepPrint == 0:

            # Restore best model on final loop
            if i == (trainSteps + 1):
                net = best_net

            # Select indices for progress report
            train_batch = random.sample(range(train_seq.shape[0]), stepBatch)
            test_batch = random.sample(range(test_seq.shape[0]), stepBatch)

            # Train and test binding predictions
            train_prediction = torch.squeeze(net(train_seq[train_batch]))
            test_prediction = torch.squeeze(net(test_seq[test_batch]))

            # Train and test accuracies
            train_accuracy = torch.abs(train_prediction - train_data[train_batch])
            train_accuracy = len(torch.nonzero(train_accuracy < tolerance)) / len(train_accuracy)
            test_accuracy = torch.abs(test_prediction - test_data[test_batch])
            test_accuracy = len(torch.nonzero(test_accuracy < tolerance)) / len(test_accuracy)

            # Train and test error
            train_error = ((train_data[train_batch] - train_prediction).data.cpu().numpy() ** 2).mean()
            test_error = ((test_data[test_batch] - test_prediction).data.cpu().numpy() ** 2).mean()
            train_test_error[((int(i / stepPrint)) - 1), 0:] = (test_error, train_error)

            # Print out
            print('Step %5d: train|test accuracy: %.2f|%.2f' %
                  (i, train_accuracy, test_accuracy))

    # Run test set through optimized neural network
    test_prediction = torch.squeeze(net(test_seq)).data.cpu().numpy()
    test_real = test_data.data.cpu().numpy()
    correlation = np.corrcoef(test_real, test_prediction)[0, 1]
    print('Correlation Coefficient: %.3f' % correlation)

    # ~~~~~~~~~~PLOTTING~~~~~~~~~~ #
    # Extract weights from model
    if Bias_inputLayer:
        amino_layer = [net.AminoLayer.weight.data.transpose(0, 1).cpu().numpy(),
                       net.AminoLayer.bias.data.cpu().numpy()]
    else:
        amino_layer = net.AminoLayer.weight.data.transpose(0, 1).cpu().numpy()
    if Bias_HiddenLayer:
        hidden_layer = [[x.weight.data.transpose(0, 1).cpu().numpy(), x.bias.data.cpu()] for x in
                        net.HiddenLayers]
    else:
        hidden_layer = [[x.weight.data.transpose(0, 1).cpu().numpy()] for x in net.HiddenLayers]
    output_layer = [net.OutputLayer.weight.data.transpose(0, 1).cpu().numpy(),
                    net.OutputLayer.bias.data.cpu().numpy()]
    plt.ioff()

    # Turn off interactive mode

    # Scatter plot of predicted vs real
    # calculate the point density
    test_xy = np.vstack([test_real, test_prediction])
    z = gaussian_kde(test_xy)(test_xy)  # it first calculates the pdf and then find the probability of a data point

    fig1, ax = plt.subplots()
    ax.scatter(test_real, test_prediction, c=z, s=5, edgecolor='')
    plt.plot([min(test_real), max(test_real)],
             [min(test_real), max(test_real)], color='k')
    if not dataShift:
        plt.xlabel('Measured', fontsize=15)
        plt.ylabel('Predicted', fontsize=15)
    else:
        plt.xlabel('log10(Measured)', fontsize=15)
        plt.ylabel('log10(Predicted)', fontsize=15)
    plt.title('Correlation Coefficient: %.3f' % correlation, fontsize=15)

    # Amino acid similarity matrix
    if Bias_inputLayer:
        amino_similar = np.linalg.norm(amino_layer[0], axis=1)
        amino_similar = np.array([aminoEigen * [magnitude] for magnitude in amino_similar])
        amino_similar = np.dot((amino_layer[0] / amino_similar),
                               np.transpose(amino_layer[0] / amino_similar))
    else:
        amino_similar = np.linalg.norm(amino_layer, axis=1)
        amino_similar = np.array([aminoEigen * [magnitude] for magnitude in amino_similar])
        amino_similar = np.dot((amino_layer / amino_similar), np.transpose(amino_layer / amino_similar))

    #  Plot amino acid similarity matrix
    fig2 = plt.matshow(amino_similar, cmap='coolwarm')
    plt.xticks(range(len(aminoAcids)), aminoAcids)
    plt.yticks(range(len(aminoAcids)), aminoAcids)
    plt.colorbar()
    plt.clim(-1, 1)

    # Scatter plot of error for train and test set
    fig3 = plt.figure()
    ax1 = fig3.add_subplot(111)
    ax1.scatter(np.arange(0, trainSteps, stepPrint), train_test_error[0:, 0], color='blue',
                label='test_batch', s=3)
    plt.scatter(np.arange(0, trainSteps, stepPrint), train_test_error[0:, 1], color='red',
                label='train_batch', s=3)
    plt.xlabel('iteration no', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title('MSE over Iteration_Sample: %.f' % sample, fontsize=15)
    plt.legend()

    # Show figures
    if showFigure:
        plt.ion()
        plt.show()

    # ~~~~~~~~~~SAVE MODEL~~~~~~~~~~ #
    # Save weights and biases for disease prediction
    if weightSave:

        # Create path to date folder
        date = sorted(os.listdir(weightFolder))[-1]
        date_folder = weightFolder + '/' + date

        # Create path to run folder
        old_runs = sum(('Run' in x for x in os.listdir(date_folder)))
        run_folder = weightFolder + '/' + date + '/Run' + str(old_runs)

        # Create directory
        hw_hl_folder = '/HL' + str(abs(hiddenLayers)) + 'HW' + str(abs(hiddenWidth))
        # Create directory to save the parameter for all of the replicates
        param_directory = run_folder + '/Sample' + str(abs(sample)) + hw_hl_folder
        directory = param_directory + '/Replicate' + str(abs(replicate_num))
        os.makedirs(directory)

        # Save weights and biases to csv files
        if Bias_inputLayer:
            np.savetxt(directory + '/W1.txt', amino_layer[0], delimiter=',')
            np.savetxt(directory + '/B1.txt', amino_layer[1], delimiter=',')
            # np.savetxt(directory + '/B1.txt', net.AminoLayer.bias.data.numpy(), delimiter=',')
        else:
            np.savetxt(directory + '/W1.txt', amino_layer, delimiter=',')
        for (m, n) in enumerate(hidden_layer):
            np.savetxt(directory + '/W' + str(m + 2) + '.txt', n[0], delimiter=',')
            if Bias_HiddenLayer:
                np.savetxt(directory + '/B' + str(m + 2) + '.txt', n[1], delimiter=',')
        np.savetxt(directory + '/WF.txt', output_layer[0], delimiter=',')
        np.savetxt(directory + '/BF.txt', output_layer[1], delimiter=',')

        # Save correlation coefficient to file
        with open(directory + '/CORR.txt', 'w') as file:
            file.write(str(correlation))

        # Save parameter settings
        with open(param_directory + '/Parameters.txt', 'w') as file:
            file.write('#~~~ARCHITECTURE~~~#\n')
            file.write(str(net))
            file.write('\n\n#~~~PARAMETERS~~~#\n')
            for m, n in paramDict.items():
                file.write(str(m) + ': ' + str(n) + '\n')

        # Save figures
        fig1.savefig(directory + '/Correlation.png', bbox_inches='tight')
        fig2.figure.savefig(directory + '/Similarity.png', bbox_inches='tight')
        fig3.savefig(directory + '/Train_Test_Error.png', bbox_inches='tight', dpi=1200)

        # Save model
        torch.save(net.state_dict(), directory + '/Model.pth')

    return correlation


if __name__ == '__main__':

    # Generate file structure
    if weightSave:

        # Create parent folder
        if not os.path.exists(weightFolder):
            os.makedirs(weightFolder)

            # Create date folder
        currentDate = datetime.datetime.today().strftime('%Y-%m-%d')
        dateFolder = weightFolder + '/' + currentDate
        if not os.path.exists(dateFolder):
            os.makedirs(dateFolder)

        # Create run folder
        oldRuns = sum(('Run' in x for x in os.listdir(dateFolder)))
        runFolder = weightFolder + '/' + currentDate + '/Run' + str(oldRuns + 1)
        os.makedirs(runFolder)

    # Training mode
    start = time.time()
    if trainingMode == 'avg':
        i = 1
        while i <= len(HL_array):
            hl = HL_array[i-1]
            params = []
            for j in HW_array:
                for k in range(epoch):
                    params.append((hl, j, -1, k))
            pool = Pool(multiprocessing.cpu_count())
            result = pool.map(training, params)
            # # extract the correlation from result and save it in Corr_array
            slice_result = []
            for value in result:
                slice_result.append(value)
            slice_result = np.reshape(slice_result, (len(HW_array), epoch))
            for j, slice_value in enumerate(slice_result):
                Corr_array[i-1][j][:] = slice_value
            i = i+1
    elif type(trainingMode) == int:
        i = 1
        while i <= len(HL_array):
            hl = HL_array[i-1]
            params = []
            for j in HW_array:
                for k in range(epoch):
                    params.append((hl, j, trainingMode, k))
            pool = Pool(multiprocessing.cpu_count())
            result = pool.map(training, params)
            # # extract the correlation from result and save it in Corr_array
            slice_result = []
            for value in result:
                slice_result.append(value)
            slice_result = np.reshape(slice_result, (len(HW_array), epoch))
            for j, slice_value in enumerate(slice_result):
                Corr_array[i-1][j][:] = slice_value
            i = i+1
        print('Invalid value for trainingMode! Must be \'avg or int\'')

    date = sorted (os.listdir (weightFolder))[-1]
    date_folder = weightFolder + '/' + date + '/'
    old_runs = sum (('Run' in x for x in os.listdir (date_folder)))
    run_folder = weightFolder + '/' + date + '/Run' + str (old_runs) + '/Sample' + str (abs (sample))
    if weightSave:
        with open (run_folder + new_file + '.txt', 'w') as outfile:
            outfile.write ('# Array shape: {0}\n'.format (Corr_array.shape))
            for data_slice in Corr_array:
                np.savetxt (outfile, data_slice, fmt='%.3f')
                outfile.write ('# with next hidden layer\n')

    mean_correlation = np.zeros ((len (HL_array), len (HW_array)))
    error_array = np.zeros ((len (HL_array), len (HW_array)))
    for HL_index, HiddenL_array in enumerate (Corr_array):
        slice_mean = np.mean (HiddenL_array, axis=1)
        slice_stdv = np.std (HiddenL_array, axis=1)
        mean_correlation[HL_index][:] = slice_mean
        error_array[HL_index][:] = slice_stdv

    mean_trans = np.transpose (mean_correlation)

    # plot average correlation of different hidden units for different hidden layers
    import matplotlib.pyplot as plt
    import matplotlib
    plt.ioff()
    fig1 = plt.figure()
    #colormap = plt.cm.gist_ncar
    # plt.gca ().set_prop_cycle (plt.cycler ('color', plt.cm.jet (np.linspace (0, 1, len(HW_array)))))

    colormap = plt.get_cmap('gist_rainbow')
    colors = [colormap(1. * i / len(HW_array)) for i in range(len(HW_array))]
    plt.gca().set_prop_cycle('color', colors)

    labels = []
    for i in range (0, len (HW_array)):
        plt.errorbar (HL_array, mean_trans[i], np.transpose (error_array)[i], linestyle='--',
                      marker='o', capsize=3.5)
        #plt.xlabel ('No of Hidden Layers', fontsize=16)
        #plt.ylabel ('Pearson Correlation Coefficient', fontsize=16)
        plt.xlim([0, 11])
        #plt.xticks([np.arange(1, len(HL_array) + 1)])
        #plt.title ('Correlation Coefficient vs No of Hidden Layers', fontsize=12)
        labels.append ('HW %i' % HW_array[i])
        plt.legend (labels, fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    fancybox=False, shadow=False, ncol=4, labelspacing=0.5, handlelength=1, handletextpad=-0.05)
        # plt.legend(labels, fontsize= 14, ncol=1, loc=6, bbox_to_anchor=[1, 0.5], columnspacing=1.0, labelspacing=1,
        #              handletextpad=0.0, handlelength=2, fancybox=False, shadow=False, borderaxespad=0.)
        #             mode='expand')
        matplotlib.rc('xtick', labelsize=18)
        matplotlib.rc('ytick', labelsize=18)

        # Now save the figures
    if weightSave:
        manager = plt.get_current_fig_manager()
        #manager.window.showMaximized()  # now maximize the window
        fig1.savefig(run_folder + '/CorrelationVsHL.png', ext='png', dpi=800)

    end = time.time()
    print(f'\nTime to complete:{end-start:.2f}s\n')
    plt.show()

# reload the correlation array from the samples that were used for different no of descriptors
if load_array:
    Corr_array = np.loadtxt(input_file)
    Corr_array = Corr_array.reshape(len(HL_array), len(HW_array), epoch)
