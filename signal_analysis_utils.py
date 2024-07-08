import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def add_to_csv(file_name, data):
    """
    Add data to a csv file
    file_name: name of the csv file
    data: list of data to be added to the csv file format [f, mse_fasrc, mse_scipy, mae_fasrc, mae_scipy]
    """
    # add_to_csv("mse_mae.csv", [f, mse_fasrc, mse_scipy, mae_fasrc, mae_scipy])
    with open("csv/" + file_name, 'a') as f:
        f.write(','.join(map(str, data)) + '\n')

def plot_mse(csv):
    """
    Plot the MSE from a csv file
    csv: name of the csv file (format: f, mse_fasrc, mse_scipy)
    """
    df = pd.read_csv("csv/" + csv)
    f = df['f']
    mse_fasrc = df['mse_fasrc']
    mse_scipy = df['mse_scipy']

    plt.figure(figsize=(4.5, 3.7))
    plt.plot(f, mse_fasrc, 'y-', label='MSE FASRC')
    plt.plot(f, mse_scipy, 'p-', label='MSE SCIPY')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Error')
    plt.legend()
    csv = csv.split(".")[0]
    plt.savefig("figs/mse.pgf")
    plt.close()

def plot_mae(csv):
    """
    Plot the MAE from a csv file
    csv: name of the csv file (format: f, mae_fasrc, mae_scipy)
    """
    df = pd.read_csv("csv/" + csv)
    f = df['f']
    mae_fasrc = df['mae_fasrc']
    mae_scipy = df['mae_scipy']

    plt.figure(figsize=(4.5, 3.7))
    plt.plot(f, mae_fasrc, 'b-', label='MAE FASRC')
    plt.plot(f, mae_scipy, 'g-', label='MAE SCIPY')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Error')
    plt.legend()
    csv = csv.split(".")[0]
    plt.savefig("figs/mae.pgf")
    plt.close()

def plot_mse_mae(csv):
    """
    Plot the MSE and MAE from a csv file
    csv: name of the csv file (format: f, mse_fasrc, mse_scipy, mae_fasrc, mae_scipy)
    """
    df = pd.read_csv("csv/" + csv)
    f = df['f']
    mse_fasrc = df['mse_fasrc']
    mse_scipy = df['mse_scipy']
    mae_fasrc = df['mae_fasrc']
    mae_scipy = df['mae_scipy']

    plt.figure(figsize=(4.5, 3.7))
    plt.plot(f, mse_fasrc, 'b-', label='MSE FASRC')
    plt.plot(f, mse_scipy, 'g-', label='MSE SCIPY')
    plt.plot(f, mae_fasrc, 'r-', label='MAE FASRC')
    plt.plot(f, mae_scipy, 'y-', label='MAE SCIPY')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Error')
    plt.legend()
    csv = csv.split(".")[0]
    plt.savefig("figs/" + csv + ".pgf")
    plt.close()

def clear_csv(file_name):
    """
    Clear a csv file
    """
    with open("csv/" + file_name, 'w') as f:
        f.write("f,mse_fasrc,mse_scipy,mae_fasrc,mae_scipy\n")

def mae(signal1, signal2):
    """
    Mean absolute error calculation between two signals.
    """
    return np.abs(np.subtract(signal1, signal2)).mean()

def mse(signal1, signal2):
    """
    Mean squared error calculation between two signals.
    """
    return np.square(np.subtract(signal1, signal2)).mean()

def plot_single(s, fs, title):
    """
    Plot one signal
    """
    t = np.arange(0, len(s)) / fs
    plt.plot(t, s, 'b-')
    plt.stem(t, s, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.savefig("figs/" + title + ".pgf")
    plt.close()


def plot_double(s1, s2, fs1, fs2, label1 = "Signal 1" , label2 = "Signal 2",description="" ):
    """
    Plot two signals with different or same sampling rates
    """
    t1 = np.arange(0, len(s1)) / fs1
    t2 = np.arange(0, len(s2)) / fs2
    plt.figure(figsize=(4.5, 3.7))
    plt.stem(t1, s1, linefmt='b-', markerfmt='b.', basefmt='r-', label=label1)
    plt.stem(t2, s2, linefmt='g-', markerfmt='g.', basefmt='r-', label=label2)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    save_file = "figs/farrow_output_" + description + ".pgf"
    plt.savefig(save_file)
    plt.close()
