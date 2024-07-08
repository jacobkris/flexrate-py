import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import fasrc
import signal_analysis_utils as sau
import time

plt.rcParams['axes.grid'] = True # set all grids to true


def test(f, num_periods, farrow_coeffs):
    print("--------------------------------------------------")
    print("Testing for frequency: ", f, "Hz")
    fs = 44100
    fs_new = 48000
    periods = num_periods
    duration = periods * 1/f 
    print("Input sampling rate: ", fs, "Hz")
    print("Output sampling rate: ", fs_new , "Hz")
    n = int(duration * fs) # number of samples
    t = np.linspace(0, duration, n, endpoint=False)
    n_new = int(duration * fs_new)
    t_new = np.linspace(0, duration, n_new, endpoint=False)

    input_signal = 10000 * np.sin(2 * np.pi * f * t) 
    input_signal = input_signal.astype(np.int32)
    output_signal_ideal = 10000 * np.sin(2 * np.pi * f * t_new)

    output_signal_scipy = signal.resample(input_signal, n_new, domain='time')

    output_signal_fasrc = fasrc.fasrc(input_signal, fs, fs_new, farrow_coeffs)

    print("Input signal length: ", len(input_signal))
    print("Output signal length: ", len(output_signal_ideal))
    print("Output Farrow signal length: ", len(output_signal_fasrc))
    
    sau.plot_double(input_signal, output_signal_fasrc, fs, fs_new, label1="Input Signal", label2="Farrow Interpolated Output", description="fasrc_" + str(f) + "_Hz")
    sau.plot_double(input_signal, output_signal_ideal, fs, fs_new, label1="Input Signal", label2="Ideal Interpolated Output", description="ideal_" + str(f) + "_Hz")
    sau.plot_double(input_signal, output_signal_scipy, fs, fs_new, label1="Input Signal", label2="Scipy Interpolated Output", description="scipy_" + str(f) + "_Hz")
    
    mse_fasrc = sau.mse(output_signal_ideal, output_signal_fasrc)
    mse_scipy = sau.mse(output_signal_ideal, output_signal_scipy)
    mae_fasrc = sau.mae(output_signal_ideal, output_signal_fasrc)
    mae_scipy = sau.mae(output_signal_ideal, output_signal_scipy)

    # add to csv
    sau.add_to_csv("mse_mae.csv", [f, mse_fasrc, mse_scipy, mae_fasrc, mae_scipy])

    print("--------------------------------------------------")


# main

# clear the csv file
sau.clear_csv("mse_mae.csv")

freqs = [500, 1000, 2000, 3000, 4000, 5000, 6000, 6800, 8000]
periods = [1, 1, 2, 2, 2, 2, 2, 2, 2]
farrow_coeffs = np.array([
    [-8.57738278e-3, 7.82989032e-1, 7.19303539e+0, 6.90955718e+0, -2.62377450e+0, -6.85327127e-1, 1.44681608e+0, -8.79147907e-1, 7.82633997e-2, 1.91318985e-1, -1.88573400e-1, 6.91790782e-2, 3.07723786e-3, -6.74800912e-3],
    [2.32448021e-1, 2.52624309e+0, 7.67543936e+0, -8.83951796e+0, -5.49838636e+0, 6.07298348e+0, -2.16053205e+0, -7.59142947e-1, 1.41269409e+0, -8.17735712e-1, 1.98119464e-1, 9.15904145e-2, -9.18092030e-2, 2.74136108e-2],
    [-1.14183319e+0, 6.86126458e+0, -6.86015957e+0, -6.35135894e+0, 1.10745051e+1, -3.34847578e+0, -2.22405694e+0, 3.14374725e+0, -1.68249886e+0, 2.54083065e-1, 3.22275037e-1, -3.04794927e-1, 1.29393976e-1, -3.32026332e-2],
    [1.67363115e+0, -2.93090391e+0, -1.13549165e+0, 5.65274939e+0, -3.60291782e+0, -6.20715544e-1, 2.06619782e+0, -1.42159644e+0, 3.75075865e-1, 1.88433333e-1, -2.64135123e-1, 1.47117661e-1, -4.71871047e-2, 1.24921920e-2]
]) / 12.28

for i in range(len(freqs)):
    test(freqs[i], periods[i], farrow_coeffs)

sau.plot_mse("mse_mae.csv")
sau.plot_mae("mse_mae.csv")

N = 1024
w = np.linspace(0, np.pi, N)
H = np.zeros(N, dtype=complex)

for i in range(4):
    H += np.polyval(farrow_coeffs[i][::-1], np.exp(-1j * w))

# plot the magnitude response
plt.figure(figsize=(4.5, 3.7))
plt.plot(w / np.pi, 20 * np.log10(np.abs(H)))
plt.xlabel(r'Normalized Frequency ($\times \pi$ rad/sample)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.savefig("figs/farrow_mag_response.pgf")
plt.close()

# plot the phase response
plt.figure(figsize=(4.5, 3.7))
plt.plot(w / np.pi, np.angle(H))
plt.xlabel(r'Normalized Frequency ($\times \pi$ rad/sample)')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.savefig("figs/farrow_phase_response.pgf")
plt.close()
