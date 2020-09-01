# This program was jointly written by Robayet chowdhury and Dr. Alexandar Taguchi


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
import time
import argparse
import datetime
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.stats import gaussian_kde
import numpy as np
import os
import pandas as pd
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

# from datetime import datetime
# startTime = datetime.now()
# import time
# startTime = time.time()


# Import parameter settings
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
# filename1 = 'sequence_data_NIBIB_Dengue_ML_CTSSeraCare_mod_CV317-Jul-2020-00-08.csv'
# filename1 = 'sequence_data_NIBIB_WNV_ML_mod_CV315-Jul-2020-23-57.csv'
# filename1 = 'sequence_data_NIBIB_HCV_ML_mod_CV317-Jul-2020-16-50.csv'
filename1 = 'sequence_data_NIBIB_HBV_ML_mod_CV316-Jul-2020-00-01.csv'
filename2 = 'sequence_data_NIBIB_Chagas_ML_mod_CV316-Jul-2020-00-02.csv'
#filename2 = 'sequence_data_NIBIB_Normal_ML_mod_CV315-Jul-2020-23-50.csv'
hiddenLayers = 2  # number of hidden layer2 - default: 20000
hiddenWidth = 250  # width of hidden layers - default: 100
trainingMode = 'all'  # train all ('all'), individuals (1, 2, 3...), r average ('avg')
trainSteps = 20000  # number of training steps - default: 20000
weightFolder = 'NIBIB_HBVvsChagas_CV3_specificity'  # name of folder for saving weights and biases
weightSave = True  # save weights to file - default: False

# ~~~~~~~~~~ADDITIONAL PARAMETERS~~~~~~~~~~ #
aminoAcids = 'ADEFGHKLNPQRSVWY'  # letter codes - default: 'ADEFGHKLNPQRSVWY'
batch = 100  # batch size for training - default: 100
dataSaturation = 0.995  # saturation threshold for training - default: 0.995
dataShift = True  # added to data before log10 (False for linear) - default: 100
gpuTraining = False  # allow for GPU training - default: False
learnRate = 0.001  # magnitude of gradient descent step - default: 0.001
minResidues = 3  # minimum sequence length - default: 3
stepPrint = 200  # step multiple at which to print progress - default: 100
stepBatch = 1000  # batch size for train/test progress report - default: 1000
tolerance = 0.1  # bin offset that still counts as a hit - default: 0.2
trainFinal = 10  # final optimization steps on all training data - default: 10
trainFraction = 0.9  # fraction of non-saturated data for training - default: 0.9
uniformBGD = True  # uniform batch gradient descent - default: True
weightRegularization = 0  # least-squares weight regularization - default: 0

# Replace parameters with imported settings
if paramImport:
    globals().update(paramImport)

# Store parameter settings
paramScope = [x for x in dir() if x not in moduleScope + ['moduleScope']]
paramLocals = locals()
paramDict = {x: paramLocals[x] for x in paramScope}

data1_seq = pd.read_csv(filename1, header=None)
data2_seq = pd.read_csv(filename2, header=None)
# combine these two data sets
data_seq = pd.DataFrame(np.column_stack((data1_seq, data2_seq.iloc[:, 1:])))
total_samples = data_seq.shape[1] - 1
total_samples1 = data1_seq.shape[1] - 1
total_samples2 = data2_seq.shape[1] - 1
# select samples and peptides for test set randomly
random.seed(5)  # random.seed() with a fixed number will allow to generate the same random numbers
if np.array_equal(data1_seq.iloc[:, 0], data2_seq.iloc[:, 0]):
    total_peptides = data1_seq.shape[0]  # total number of peptides in the data set
    testSet_index = sorted(random.sample(list(range(0, total_peptides)), int((1 - trainFraction) * total_peptides)))
    # print(testSet_index[0:10])
    #  randomly pick a set of  peptides for a test set
    data_index = np.zeros(total_peptides, dtype='int64')
    # create an array of binary numbers to divide the data set into test
    # paramDict.update(total_samples=total_samples)
    paramDict.update(total_peptides=total_peptides)
    #  create a boolean array for the train and test set
    data_index = np.zeros(total_peptides, dtype='int64')
    for j, i in enumerate(testSet_index):
        data_index[i] = 1  # select a peptide to include into the test set
    # # create  zero matrices to save measured and predicted bindin♫g data
    measured_test = np.zeros((int((1 - trainFraction) * total_peptides), total_samples))
    predicted_test = np.zeros((int((1 - trainFraction) * total_peptides), total_samples))
else:
    print('two data sets have different no. of peptides')
Case1_Case2 = 'HBVvsChagas'


def training(sample):
    print('running sample: %.f' % sample)
    # paramDict.update(trainingMode=sample)

    # Import matplotlib into new environment
    import matplotlib.pyplot as plt

    # Import csv (first column is the sequences followed by the binding data)
    data = data_seq
    data = pd.DataFrame(np.column_stack((data, data_index)))  # add the data_index at the end of the dataset

    # Extract column for assigning training and testing sets
    data_split = np.array([], dtype=int)
    if set(data.iloc[:, -1]) == {0, 1}:
        data_split = data.iloc[:, -1].values
        data = data.iloc[:, :-1]

    # Check for bad sequence entries
    data[0].replace(re.compile('[^' + aminoAcids + ']'), '', inplace=True)

    # Remove trailing GSG from sequence
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

    # Train all, one, or averaged data
    if sample > 0:
        data = data[sample].values
    else:
        data = np.mean(data.loc[:, 1:].values, axis=1)

    data = data.astype(np.float64)

    # Apply baseline shift and base-10 logarithm , and update dataShift
    if dataShift is True and total_samples >= 1:
        print('subtracting min value and adding 1')
        data = np.log10(data - (min(data) - 1))
        # paramDict.update(dataShift=dataShift)
    elif dataShift is float or int and total_samples >= 1:
        print('adding dataShift')
        data = np.log10(data + dataShift)

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

    print('no of peptides in a training set:', train_xy.shape[0])

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
    # linearize the data from measured and predicted test set
    test_real_linear = 10 ** test_real
    test_prediction_linear = 10 ** test_prediction

    # ~~~~~~~~~~PLOTTING~~~~~~~~~~ #
    # Extract weights from model
    if Bias_inputLayer:
        amino_layer = [net.AminoLayer.weight.data.transpose(0, 1).cpu().numpy(),
                       net.AminoLayer.bias.data.cpu().numpy()]
    else:
        amino_layer = net.AminoLayer.weight.data.transpose(0, 1).cpu().numpy()
    if Bias_HiddenLayer:
        hidden_layer = [[x.weight.data.transpose(0, 1).cpu().numpy(), x.bias.data.cpu()] for x in net.HiddenLayers]
    else:
        hidden_layer = [[x.weight.data.transpose(0, 1).cpu().numpy()] for x in net.HiddenLayers]
    output_layer = [net.OutputLayer.weight.data.transpose(0, 1).cpu().numpy(),
                    net.OutputLayer.bias.data.cpu().numpy()]
    # Turn off interactive mode
    plt.ioff()

    # Scatter plot of predicted vs real
    # calculate the point density
    test_xy = np.vstack([test_real, test_prediction])
    z = gaussian_kde(test_xy)(test_xy)  # it first calculates the pdf and then find the probability of a data point
    fig1, ax = plt.subplots()
    ax.scatter(test_real, test_prediction, c=z, s=5, edgecolor='')
    leastSqr_fit = np.polyfit(test_real, test_prediction, 1)  # least squares fit to data
    p = np.poly1d(leastSqr_fit)  # 1 dimensional polynomial class
    plt.plot(test_real, p(test_real), lw=1, color='k')  # add a trend line on the scatter plot
    # plt.plot([min(test_real), max(test_real)],
    #          [min(test_real), max(test_real)], color='k')
    if not dataShift:
        plt.xlabel('Measured', fontsize=15)
        plt.ylabel('Predicted', fontsize=15)
    else:
        plt.xlabel('log10(Measured)', fontsize=15)
        plt.ylabel('log10(Predicted)', fontsize=15)
    ax.text(0.05, 0.95, 'R=%.3f' % correlation, transform=ax.transAxes,
            verticalalignment='top', fontsize=15)  # add correlation coefficient as a title inside the plot

    # Amino acid similarity matrix
    if Bias_inputLayer:
        amino_similar = np.linalg.norm(amino_layer[0], axis=1)
        amino_similar = np.array([aminoEigen * [magnitude] for magnitude in amino_similar])
        amino_similar = np.dot((amino_layer[0] / amino_similar), np.transpose(amino_layer[0] / amino_similar))
    else:
        amino_similar = np.linalg.norm(amino_layer, axis=1)
        amino_similar = np.array([aminoEigen * [magnitude] for magnitude in amino_similar])
        amino_similar = np.dot((amino_layer / amino_similar), np.transpose(amino_layer / amino_similar))

    # Plot amino acid similarity matrix
    fig2 = plt.matshow(amino_similar, cmap='coolwarm')
    plt.xticks(range(len(aminoAcids)), aminoAcids)
    plt.yticks(range(len(aminoAcids)), aminoAcids)
    plt.colorbar()
    plt.clim(-1, 1)

    # Scatter plot of error for train and test set
    fig3 = plt.figure()
    ax1 = fig3.add_subplot(111)
    ax1.scatter(np.arange(0, trainSteps, stepPrint), train_test_error[0:, 0], color='blue',
                label='test_batch', s=5)
    plt.scatter(np.arange(0, trainSteps, stepPrint), train_test_error[0:, 1], color='red',
                label='train_batch', s=5)
    plt.xlabel('iteration no', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title('MSE over Iteration_Sample: %.f' % sample, fontsize=15)
    plt.legend()

    # Show figures
    if not weightSave:
        plt.ion()
        # plt.show ()

    # ~~~~~~~~~~SAVE MODEL~~~~~~~~~~ #
    # Save weights and biases for disease prediction
    if weightSave:

        # Create path to date folder
        date = sorted(os.listdir(weightFolder))[-1]
        date_folder = weightFolder + '/' + date

        # Create path to run folder
        old_runs = sum(('Run' in x for x in os.listdir(date_folder)))
        run_folder = weightFolder + '/' + date + '/Run' + str(old_runs)

        # Create sample folder
        directory = run_folder + '/Sample' + str(abs(sample))
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
        with open(run_folder + '/Parameters.txt', 'w') as file:
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
        torch.save(net.state_dict(), directory + '/Model.path')

    # return test_real, test_prediction
    return test_real_linear, test_prediction_linear
    # Since Z score fitting does not use log10 scale , use linear scale to keep the result consistent with the Z score
    # fitting


if __name__ == '__main__':
    # Generate file structure
    # if weightSave:
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
    if not weightSave:
        # Save parameter settings
        with open(runFolder + '/Parameters.txt', 'w') as file:
            # Save parameter settings    file.write('#~~~ARCHITECTURE~~~#\n')
            file.write('\n\n#~~~PARAMETERS~~~#\n')
            for m, n in paramDict.items():
                file.write(str(m) + ': ' + str(n) + '\n')

    # Training mode
    start = time.time()
    if trainingMode == 'all':
        pool = Pool()
        result = pool.map(training, range(1, total_samples + 1))
        for i, slice_result in enumerate(result):
            for ii, slice_bindingData in enumerate(slice_result):
                if ii == 0:
                    measured_test[:, i] = slice_bindingData
                else:
                    predicted_test[:, i] = slice_bindingData

    elif trainingMode == 'avg':
        training(-1)
    elif type(trainingMode) == int:
        training(trainingMode)
    else:
        print('Invalid value for trainingMode! Must be \'all\', \'avg\', or 1, 2, 3...')

    if trainingMode == 'all':
        # plot Z scores for test set from measured and prediction data
        # calculate mean and standard deviation from each data set
        mean_measured_test_data1 = np.mean(measured_test[:, 0:total_samples1], axis=1)
        mean_measured_test_data2 = np.mean(measured_test[:, total_samples1:], axis=1)
        std_measTest1 = np.std(measured_test[:, 0:total_samples1], axis=1)
        std_measTest2 = np.std(measured_test[:, total_samples1:], axis=1)
        # # now calculate Z scores between two cases
        Z_measTest = (mean_measured_test_data1 - mean_measured_test_data2) / np.sqrt(std_measTest1**2 + std_measTest2**2)
        #Z_measTest = (mean_measured_test_data1 - mean_measured_test_data2) / np.sqrt((std_measTest1 ** 2 / total_samples1) + (std_measTest2 ** 2 / total_samples2))
        mean_predicted_test_data1 = np.mean(predicted_test[:, 0:total_samples1], axis=1)
        mean_predicted_test_data2 = np.mean(predicted_test[:, total_samples1:], axis=1)
        std_predTest1 = np.std(predicted_test[:, 0:total_samples1], axis=1)
        std_predTest2 = np.std(predicted_test[:, total_samples1:], axis=1)
        Z_predTest = (mean_predicted_test_data1 - mean_predicted_test_data2) / np.sqrt(std_predTest1 ** 2 + std_predTest2 ** 2)
        #Z_predTest = (mean_predicted_test_data1 - mean_predicted_test_data2) / np.sqrt((std_predTest1 ** 2 / total_samples1) + (std_predTest2 ** 2 / total_samples2))
        # # calculate correlation coefficient between Z scores of measured and predicted cases
        correlation_Z = np.corrcoef(Z_measTest, Z_predTest)[0, 1]
        # combine measured and predicted Z scores
        meas_pred_Z = np.vstack([Z_measTest, Z_predTest])
        # Save the Z scores from measured and predicted binding values from both cases
        #np.savetxt(runFolder + '/Z scores_meas_pred.txt', meas_pred_Z, delimiter='')
        import matplotlib.pyplot as plt

        # first calculate the pdf and then find the probability of a data point
        density_Z = gaussian_kde(meas_pred_Z)(meas_pred_Z)
        # set up a same scale for both axis of a plot to be created below
        xmin = Z_measTest.min()
        xmax = Z_measTest.max()
        ymin = Z_predTest.min()
        ymax = Z_predTest.max()
        if xmin < ymin:
            X_min = round(xmin - 0.5)
            if X_min % 2 != 0:  # this will force the scale to use an even interval
                X_min = X_min - 1
            Y_min = X_min
        else:
            X_min = round(ymin - 0.5)
            if X_min % 2 != 0:  # this will force the scale to use an even interval
                X_min = X_min - 1
            Y_min = X_min
        if xmax > ymax:
            X_max = round(xmax + 0.5)
            if X_max % 2 != 0:  # this will force the scale to use an even interval
                X_max = X_max + 1
            Y_max = X_max
        else:
            X_max = round(ymax + 0.5)
            if X_max % 2 != 0:  # this will force the scale to use an even interval
                X_max = X_max + 1
            Y_max = X_max
        # plot the scatter plot with density
        fig1, ax = plt.subplots()
        ax.scatter(Z_measTest, Z_predTest, c=density_Z, s=5, edgecolor='')
        leastSqr_fit = np.polyfit(Z_measTest, Z_predTest, 1)  # least squares fit to data
        p = np.poly1d(leastSqr_fit)  # 1 dimensional polynomial class
        plt.plot(Z_measTest, p(Z_measTest), lw=1, color='k')  # add a trend line on the scatter plot
        #plt.xlabel('Measured Z score', fontsize=15)
        #plt.ylabel('Predicted Z score', fontsize=15)
        ax.text(0.05, 0.95, 'R=%.3f' % correlation_Z, transform=ax.transAxes,
                verticalalignment='top', fontsize=15)  # add correlation coefficient as a title inside the plot
        if X_max - X_min > 10:
            plt.xticks(np.arange(X_min, X_max + 2, 2))
            plt.yticks(np.arange(Y_min, Y_max + 2, 2))
        else:
            plt.xticks(np.arange(X_min, X_max + 1, 1))
            plt.yticks(np.arange(Y_min, Y_max + 1, 1))
        plt.xlim(X_min, X_max)
        plt.ylim(Y_min, Y_max)
        fig1.savefig(runFolder + '/' + Case1_Case2 + '.png', bbox_to_anchor=(0.5, 0.5), ext='png', dpi=1200)
        end = time.time()
    print(f'\nTime to complete:{(end - start) / 60:.2f} min\n')
    plt.show()
