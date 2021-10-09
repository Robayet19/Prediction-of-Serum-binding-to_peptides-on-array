# This program was originally written by Dr. Alexandar Taguchi and later significantly modified by Robayet chowdhury
# The program evaluates the performance (Pearson Correlation Coefficient (R)) of a NN model as a function of different numbre of peptides in a training set

# ---Neural Network for Peptide Array Interpolation--- #
#
# Architecture:
#    Input: Sequence (residue number x amino acid type matrix),
#    Neural Network: 1 fully connected hidden layer to encode
#        amino acid eigenvectors , a flattening
#        operation, multiple fully connected hidden layers ,
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
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from tkinter import *  # this is required to input a file from a directory using GUI
from tkinter.filedialog import askopenfilename
import time

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
drop_prob = 0.1 # probability of removing nodes in a hidden layer -default: 0.5
filename = 'sequence_data_NIBIB_Dengue_ML_CTSSeraCare_mod_CV317-Jul-2020-00-08.csv' # import sequence and binding data
# filename = 'NIBIB_Dengue4(CTSSera)ND_Zscore_multisample_eval_CV3.cs♦v'
hiddenLayers = 2  # number of hidden layers - default: 1
hiddenWidth = 250  # width of hidden layers - default: 100
trainSteps = 20000  # number of training steps - default: 20000
trainingMode = 'all'  # train all ('all'), individuals (1, 2, 3...), or average ('avg')
weightFolder = 'NIBIB_DENV_samplefits'  # name of folder for saving weights and biases
weightSave = True  # save weights to file - default: False

# ~~~~~~~~~~ADDITIONAL PARAMETERS~~~~~~~~~~ #
aminoAcids = 'ADEFGHKLNPQRSVWY'  # letter codes - default: 'ADEFGHKLNPQRSVWY'
batch = 100  # batch size for training - default: 100
Batch_normalization = False  # perform 1D batch normalization in hidden layer/s
epoch = 5  # no of times the dataset to be run - default: 2
dataSaturation = 0.995  # saturation threshold for training - default: 0.995
dataShift = 0.01  # added to data before log10 (False for linear) - default: 100
gpuTraining = False  # allow for GPU training - default: False
learnRate = 0.001  # magnitude of gradient descent step - default: 0.001
minResidues = 3  # minimum sequence length
num_samples_used = 10  # number of samples to be used for training - default: 2
Select_pepTest = True  # select peptides for test set - default: False
stepPrint = 200  # step multiple at which to print progress - default: 100
stepBatch = 1000  # batch size for train/test progress report - default: 1000
tolerance = 0.1  # bin offset that still counts as a hit - default: 0.2
total_samples = pd.read_csv(filename, header=None).shape[1] - 1  # total number of samples in the dataset
total_peptides = pd.read_csv(filename, header=None).shape[0]  # total number of peptides in the dataset
trainFinal = 10  # final optimization steps on all training data - default: 10
trainFraction = 0.9  # fraction of non-saturated data for training - default: 0.9
uniformBGD = True  # uniform batch gradient descent - default: True
weightRegularization = 0  # least-squares weight regularization - default: 0

# Replace parameters with imported settings
if paramImport:
    globals().update(paramImport)

paramScope = [x for x in dir() if x not in moduleScope + ['moduleScope']]
paramLocals = locals()
paramDict = {x: paramLocals[x] for x in paramScope}

# Create arrays for sample indices, aminoEigen, and Correlation
train_array = (0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
if trainingMode == 'all':
    sampleIndex_array = sorted(random.sample(range(1, total_samples), num_samples_used))  # select samples randomly
    Corr_array = np.zeros((len(train_array), num_samples_used, epoch))
elif trainingMode == 'avg' or trainingMode == 1:
    Corr_array = np.zeros((len(train_array), epoch))

load_array = False  #
showFigure = False
sampleOfinterest = False  # plot results for only one sample
new_file = '/CorrVsPepNum_DENV_samples.txt'  # save the correlation coefficients in a text file


def training(params):
    """ A feed forward neural network (NN) explained below calculates binding value for a given peptide sequence. Here,
        the NN model can be trained on antibody binding data of varying number peptide sequences in multiple individuals
        or serum samples. Overall, this function will output a Pearson correlation coefficient value as a measure of the
        NN model's performance.
        
    params : it is a set of additional parameters with a shape of [3 x 1]
           params[0] {float}: number of peptide sequences to be used to train the NN model
           params[1] {int or str}: index {float} of a sample in a cohort data or an average {str} of the entire cohort  
           params[2] {int}: replicate number to train the NN model with the parameters mentioned above
           """
    trainFraction = params[0]
    trainingMode = params[1]
    rep_num = params[2]
    replicate_num = rep_num + 1
    print('Percent of total peptides: %.3f|Sample no: %.f|Replicate no: %.f' %
          (trainFraction * 100, trainingMode, replicate_num))
    if trainingMode < 0:
        sample = params[1]
        trainingMode = 'avg'
    else:
        sample = trainingMode
    # update main parameters
    paramDict.update(trainingMode=trainingMode)
    paramDict.update(trainFraction=trainFraction)
    paramDict.update(replicate_num=replicate_num)
    # Import matplotlib into new environment
    import matplotlib.pyplot as plt
    # Import csv (first column is the sequences followed by the binding data)
    data = pd.read_csv(filename, header=None)
    if Select_pepTest:
        # select a set of peptides randomly for test set
        testSet_index = np.array(
            random.sample(list(range(0, total_peptides)), int((1 - trainFraction) * total_peptides)))
        # create an array of '0's and '1's to divide the dataset into test
        data_index = np.zeros(total_peptides, dtype='int64')
        for _, i in enumerate(testSet_index):
            data_index[i] = 1  # select a peptide to include into the test set

        print('adding boolean indices for test and train set')
        data = pd.DataFrame(np.column_stack((data, data_index)))  # add the data_index at the end of the dataset

    # Extract column for assigning training and testing sets
    data_split = np.array([], dtype=int)
    if set(data.iloc[:, -1]) == {0, 1}:
        data_split = data.iloc[:, -1].values
        data = data.iloc[:, :-1]

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

    # Train all, one, or averaged data
    if sample > 0:
        data = data[sample].values
    else:
        data = np.mean(data.loc[:, 1:].values, axis=1)
    data = data.astype(np.float64)

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
          """ Fully or partially connected Feed Forward Neural Network to predict peptide binding to serum Antibodies 
              using peptide sequence similar to the one from the paper by Taguchi & Woodbury et al,Combinatorial Science(2020)
            
          Architecture: 
          Input: Peptide sequence is represented as a matrix with shape of [number of residue positions X 
                             number of amino acids] 
          Model: 
               [1] Input layer: A linear encoder creates an embedding matrix through a linear transformation of one-hot 
                                amino acid representation into a dense representation
               [2] Hidden layers : Fully or partially connected multiple hidden layers in a Feed-Forward Neural Network
               [3] Output layer: A linear regression model gives a continuous binding value for the input sequence                     
          """
        def __init__(self, width, layers, drop):
            """ Hyperparameters associated with the Neural Network

            Keyword arguments:
                    width {int}: number of hidden nodes
                    layers {int}: number of hidden layers
                    drop {float} : drop out rate
                """
            def __init__(self, width, layers, drop):
                super().__init__()

                # Network layers
                if Bias_inputLayer:
                    self.AminoLayer = nn.Linear(len(aminoAcids), aminoEigen, bias=True)
                else:
                    self.AminoLayer = nn.Linear(len(aminoAcids), aminoEigen, bias=False)
                self.Amino_BatchNorm = nn.BatchNorm1d(int(max_len * aminoEigen))  # applying batch normalization
                if Bias_HiddenLayer:
                    self.HiddenLayers = nn.ModuleList(
                        [nn.Linear(int(max_len * aminoEigen), width, bias=True)])
                    self.HiddenLayers.extend([nn.Linear(width, width, bias=True) for _ in range(layers - 1)])
                else:
                    self.HiddenLayers = nn.ModuleList(
                        [nn.Linear(int(max_len * aminoEigen), width, bias=False)])
                    self.HiddenLayers.extend(
                        [nn.Linear(width, width, bias=False) for _ in range(layers - 1)])
                self.Hidden_BatchNorm = nn.BatchNorm1d(width)  # applying batch normalization

                self.OutputLayer = nn.Linear(width, 1, bias=True)
                self.dropout = nn.Dropout(p=drop)

            def forward(self, seq):
                out = seq.view(-1, len(aminoAcids))
                out = self.AminoLayer(out)
                out = out.view(-1, int(max_len * aminoEigen))
                if Batch_normalization:
                    out = self.Amino_BatchNorm(out)  # applying batch normalization
                for j, x in enumerate(self.HiddenLayers):
                    if j !=0 :
                        out = self.Hidden_BatchNorm(out)  # applying batch normalization on the second layer and beyond
                    out = self.dropout(functional.relu(x(out)))
                out = self.OutputLayer(out)
                return out

    net = NeuralNet(width=hiddenWidth, layers=hiddenLayers, drop=drop_prob).to(device)
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
        # turn on evaluation mode to stop using dropout method
        net.eval() 

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
        # turn on training mode to use dropout method
        net.train()  

    # Run test set through optimized neural network
    net.eval()  # turn on evaluation mode to stop using dropout method
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
    fig1, ax = plt.subplots()
    if trainFraction == 0.9:
        #  calculate the point density 
        from scipy.stats import gaussian_kde
        # if you have not  imported gaussian_kde at the begining (~modules~),then import it here
        test_xy = np.vstack([test_real, test_prediction])
        z = gaussian_kde(test_xy)(test_xy)  # it first calculates the pdf and then find the proabability of a datapoint
        ax.scatter(test_real, test_prediction, c=z, s=5, edgecolor='')
    else:
        ax.scatter(test_real, test_prediction, color='b', s=1)
    leastSqr_fit = np.polyfit(test_real, test_prediction, 1)  # least squares fit to data
    p = np.poly1d(leastSqr_fit)  # 1 dimensional polynomial class
    plt.plot(test_real, p(test_real), lw=1, color='k')  # add a trend line on the scatter plot
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
        amino_similar = np.dot((amino_layer[0] / amino_similar),
                               np.transpose(amino_layer[0] / amino_similar))
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
    ax1.scatter(np.arange(0, trainSteps, stepPrint), train_test_error[0:, 0], color='blue', label='test_batch', s=5)
    plt.scatter(np.arange(0, trainSteps, stepPrint), train_test_error[0:, 1], color='red', label='train_batch', s=5)
    plt.xlabel('iteration no', fontsize=12)
    plt.ylabel('Mean Square Error', fontsize=12)
    plt.title('MSE over Iteration_Sample: %.f' % sample, fontsize=15)
    plt.legend()

    # Show figures
    # if not weightSave:
    if showFigure:
        plt.ion()
        plt.show()

    # ~~~~~~~~~~SAVE MODEL~~~~~~~~~~ #
    # Save weights and biases for disease prediction
    if weightSave:

        # Create sample folder
        # Create path to date folder
        date = sorted(os.listdir(weightFolder))[-1]
        date_folder = weightFolder + '/' + date

        # Create path to run folder
        old_runs = sum(('Run' in x for x in os.listdir(date_folder)))
        run_folder = weightFolder + '/' + date + '/Run' + str(old_runs)

        # Create directory
        sample_tf_folder = '/Sample' + str(abs(sample)) + '/Train' + str(abs(trainFraction * 100)) + 'percent'
        # Create directory to save the parameter for all of the replicates
        param_directory = run_folder + sample_tf_folder
        directory = run_folder + sample_tf_folder + '/Replicate' + str(abs(replicate_num))
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

    # Training mode
    start = time.time()
    num_workers = 25
    if trainingMode == 'all':
        i = 1
        while i <= len(train_array):
            ts = train_array[i - 1]
            params = []
            for j in sampleIndex_array:
                for k in range(epoch):
                    params.append((ts, j, k))
            pool = Pool(num_workers)
            result = pool.map(training, params)
            result = np.reshape(result, (num_samples_used, epoch))
            for j, slice_result in enumerate(result):
                Corr_array[i - 1][j][:] = slice_result
            del result
            del slice_result
            i = i + 1
    elif trainingMode == 'avg':
        i = 1
        while i <= len(train_array):
            ts = train_array[i - 1]
            params = []
            for k in range(epoch):
                params.append((ts, -1, k))
            pool = Pool(num_workers)
            result = pool.map(training, params)
            for j, slice_result in enumerate(result):
                Corr_array[i - 1][j] = slice_result
            del result
            i = i + 1

    elif type(trainingMode) == int:
        i = 1
        while i <= len(train_array):
            ts = train_array[i - 1]
            params = []
            for k in range(epoch):
                params.append((ts, trainingMode, k))
            pool = Pool(num_workers)
            result = pool.map(training, params)
            for j, slice_result in enumerate(result):
                Corr_array[i - 1][j] = slice_result
            del result
            i = i + 1
    else:
        print('Invalid value for trainingMode! Must be \'all or avg\'')

    date = sorted(os.listdir(weightFolder))[-1]
    date_folder = weightFolder + '/' + date + '/'
    old_runs = sum(('Run' in x for x in os.listdir(date_folder)))
    run_folder = weightFolder + '/' + date + '/Run' + str(old_runs)

    with open(run_folder + new_file, 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(Corr_array.shape))
        for data_slice in Corr_array:
            np.savetxt(outfile, data_slice, fmt='%.3f')
            outfile.write('# with next trainFracr\n')

    # multiple lines plot
    import matplotlib.pyplot as plt

    plt.ioff()
    fig1 = plt.figure()
    if trainingMode == 'all':
        mean_correlation = np.zeros((len(train_array), num_samples_used))
        error_array = np.zeros((len(train_array), num_samples_used))
        for train_index, ts_array in enumerate(Corr_array):
            slice_mean = np.mean(ts_array, axis=1)
            slice_stdv = np.std(ts_array, axis=1)
            mean_correlation[train_index][:] = slice_mean
            error_array[train_index][:] = slice_stdv

        mean_trans = np.transpose(mean_correlation)
        colormap = plt.get_cmap('gist_rainbow')
        colors = [colormap(1. * i / num_samples_used) for i in range(num_samples_used)]
        plt.gca().set_prop_cycle('color', colors)
        labels = []
        for i in range(0, len(sampleIndex_array)):
            plt.errorbar(np.multiply(train_array, total_peptides), mean_trans[i], np.transpose(error_array)[i],
                         linestyle='--',
                         marker='o', capsize=3.5)
            plt.xlabel('no of peptides in training set', fontsize=10)
            plt.ylabel('Pearson Correlation Coefficient', fontsize=10)
            plt.xlim([-4200, 120000])
            plt.ylim([0.75, 1])
            labels.append('S %i' % sampleIndex_array[i])
            plt.legend(labels, ncol=1, loc=6, bbox_to_anchor=[1, 0.5], columnspacing=1.0, labelspacing=1,
                       handletextpad=0.0, handlelength=2, fancybox=False, shadow=False, borderaxespad=0., mode='expand')
            # fig1.savefig(runFolder + '/CorrelationVsPepnum.png', bbox_to_anchor=(0.5, 0.5), ext='png', dpi=1000)
    elif trainingMode == 'avg' or type(trainingMode) == int:
        mean_correlation = np.mean(Corr_array, axis=1)
        error_array = np.std(Corr_array, axis=1)
        plt.errorbar(np.multiply(train_array, total_peptides), np.transpose(mean_correlation), error_array,
                     linestyle='--', marker='o', capsize=3.5)
        plt.xlabel('no of peptides in training set', fontsize=10)
        plt.ylabel('Pearson Correlation Coefficient', fontsize=10)
        plt.xlim([-4200, 120000])
        plt.ylim([0.75, 1])
        # plt.title('Correlation Coefficient vs no of total peptides in training set', fontsize=12)

    elif trainingMode == 'avg' or type(trainingMode) == int:
        mean_correlation = np.mean(Corr_array, axis=1)
        error_array = np.std(Corr_array, axis=1)
        plt.errorbar(np.multiply(train_array, total_peptides), np.transpose(mean_correlation), error_array,
                     linestyle='--',
                     marker='o', capsize=3.5)
        plt.xlabel('no of peptides in training set', fontsize=10)
        plt.ylabel('Pearson Correlation Coefficient', fontsize=10)
        plt.xlim([-4200, 120000])
        plt.ylim([0.5, 1])
        # plt.title('Correlation Coefficient vs no of total peptides in training set', fontsize=12)
        leastSqr_fit = np.polyfit (np.multiply (train_array[4:], total_peptides), mean_correlation[4:], 1)
        p = np.poly1d (leastSqr_fit)
        plt.plot (np.multiply (train_array[4:], total_peptides), p (np.multiply (train_array[4:], total_peptides)),
                  lw=1, color='k')

    # save the figure
    fig1.savefig(runFolder + '/CorrelationVsPepnum.png', bbox_to_anchor=(0.5, 0.5), ext='png', dpi=1200)

    if sampleOfinterest:
        fig2 = plt.figure()
        i = sampleIndex_array.index(sampleOfinterest)
        plt.gca().set_prop_cycle('color', [colors[i]])
        labels = []
        plt.errorbar(np.multiply(train_array, 100), mean_trans[i], np.transpose(error_array)[i], linestyle='--',
                     marker='o', capsize=3.5)
        plt.xlabel('% of total peptides in training set', fontsize=10)
        plt.ylabel('Pearson Correlation Coefficient', fontsize=10)
        plt.ylim([0.5, 1])
        plt.ylim([0.5, 1])
        # plt.title('Correlation Coefficient vs % of total peptides in training set', fontsize=12)
        labels.append(disease_name[i])
        plt.legend(labels, ncol=1, loc=6, bbox_to_anchor=[1, 0.5], columnspacing=1.0, labelspacing=1,
                   handletextpad=0.0, handlelength=2, fancybox=False, shadow=False, borderaxespad=0., mode='expand')
        plt.show()
        # save the figure
        fig2.savefig(runFolder + '/CorrelationVsPepnum_Sample' + str(abs(sampleOfinterest)) + '.png', bbox_to_anchor=
        (0.5, 0.5), ext='png', dpi=1200)

    end = time.time()
    print(f'\nTime to complete:{end - start:.2f}s\n')
    plt.show()

if load_array:
    input_file = new_file
    Corr_array = np.loadtxt(input_file)
    Corr_array = Corr_array.reshape(len(train_array), num_samples_used, epoch)
