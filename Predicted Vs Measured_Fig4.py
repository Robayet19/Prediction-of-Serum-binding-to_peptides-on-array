from scipy.stats import gaussian_kde
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

window = Tk()
window.geometry('700x600')

def open_file():
    current_directory = os.getcwd()
    directory_csvfile = filedialog.askopenfilename(title="Select a csv file", filetypes=(("csv files", "*.csv"),
                                                                                             ("allfiles", "*.*")))
    directory = os.path.split(directory_csvfile)[0]  # parse the directory where the csv file is located
    filename = os.path.split(directory_csvfile)[1]  # parse the name of the csvfile from that directory
    print('name of a csvfile =', filename)
    print('directory:', directory)
    os.chdir(directory)  # change the directory to the fastafile location
    file_data = pd.read_csv(filename, header=None) # read
    return file_data


button = Button(window, text="Click a file", command=lambda: open_file())
button.pack(side=TOP, pady=5)
greeting = Label(text="Click me")
greeting.pack()
window.title('Select a file')
#mainloop()
window.withdraw()

# extract data from open_file()
data = open_file()
disease_state = ['DENV4', 'WNV', 'HCV', 'HBV', 'Chagas', 'Uninfected']
measured_data = data.iloc[:, 0:len(disease_state)]
predicted_data = data.iloc[:, len(disease_state):]


# plot the measured and predicted data
for i in range(len(disease_state)):
    x = measured_data.iloc[1:, i].values
    y = predicted_data.iloc[1:, i].values
    x = x.astype(float)
    y = y.astype(float)
    correlation = np.corrcoef(x, y)[0, 1]
    meas_pred = np.vstack([x, y])  # merge two arrays
    z = gaussian_kde(meas_pred)(meas_pred)  # calculate density
    # plot the measured and predicted binding data with density
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=5)
    least_squareFit = np.polyfit(x, y, 1)  # least squares fit to data
    polnom_class = np.poly1d(least_squareFit)  # 1 dimensional polynomial class
    plt.plot(x, polnom_class(x), lw=1, color='k')  # add a trend line on the scatter plot
    plt.title(disease_state)
    plt.xlabel('Measured', fontsize=15)
    plt.ylabel('Predicted', fontsize=15)
    plt.xlim([-1.2, 1.4])
    plt.ylim([-1.2, 1.4])
    ax.text(0.05, 0.95, 'R=%.3f' % correlation, transform=ax.transAxes,
            verticalalignment='top', fontsize=15)  # add correlation coefficient as a title inside the plot








