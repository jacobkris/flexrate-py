# FASRC (Flexible arbitrary sampling rate converter) based on Tor Ramstad paper

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def plot_single(signal, fs, title="Signal"):
    """
    Plot one signal
    """
    t_input = np.arange(0, len(signal)) / fs
    plt.plot(t_input, signal, 'b-')
    plt.stem(t_input, signal, linefmt='b-', markerfmt='r.', basefmt='r-')
    plt.title(title)
    plt.show()

def plot_double(signal1, signal2, fs1, fs2, title1="Signal 1", title2="Signal 2"):
    """
    Plot two signals
    """
    pass

def plot_freq_response(coeffs, fs_in):
    """
    Plot frequency response of filter
    """
    # plot freq response
    w, h = signal.freqz(coeffs, fs=fs_in)
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.title('FIR filter frequency response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(which='both', axis='both')
    plt.show()

def fir_precalculate_coeffs(fs_in,fs_out,cutoff,num_taps):
    """
    FIR filter coefficients
    """
    ratio = fs_out / fs_in
    normalized_cutoff = cutoff / fs_in
    coeffs = signal.firwin(num_taps, normalized_cutoff / ratio, window='hamming')
    return coeffs

def fasrc(input_signal, fs_in, fs_out, coeffs):
    """
    Flexible ASRC
    """
    coeff_order = len(coeffs)
    input_period = 1 / fs_in 
    output_period = 1 / fs_out
    output_length = int(np.ceil(len(input_signal) * fs_out / fs_in))
    output_signal = np.zeros(output_length)

    for sample_idx in range(output_length):
        t_k = sample_idx * output_period
        n = int(t_k / input_period)
        delta_t = t_k - n * input_period
        offset = int(np.round(delta_t / input_period))
        for m in range(coeff_order):
            if 0 <= n - offset + m  < len(input_signal):
                output_signal[sample_idx] += coeffs[m] * input_signal[n - offset + m]

    return output_signal

if __name__ == '__main__':
    # test signal
    f = 440 # Hz
    fs_in = 41100 # Hz
    fs_out = 48000 # Hz
    periods = 4
    duration = periods / f
    print("Frequency of signal: ", f, "Hz")
    print("Input sampling rate: ", fs_in)
    print("Output sampling rate: ", fs_out)
    n = int(fs_in * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    input_signal = 10000 * np.sin(2 * np.pi * f * t)
    n_up = int(fs_out * duration)
    t_up = np.linspace(0, duration, n_up, endpoint=False)
    upsampled_signal = 10000 * np.sin(2 * np.pi * f * t_up)
    input_signal = input_signal.astype(np.int32)
    print("Number of samples in input signal: ", n)
    print("First 10 samples of input signal: ", input_signal[:10])
    plot_single(input_signal, fs_in, title="Input Signal")

    # filter design
    num_taps = 16
    cutoff = 20000
    coeffs = fir_precalculate_coeffs(fs_in, fs_out, cutoff, num_taps)
    plot_freq_response(coeffs, fs_in)

    output_signal = fasrc(input_signal, fs_in, fs_out, coeffs)
    output_signal = output_signal.astype(np.int32)
    print("Number of samples in output signal: ", len(output_signal))
    print("First 10 samples of output signal: ", output_signal[:10])
    plot_single(output_signal, fs_out, title="Output Signal")
    print("Number of samples in upsampled signal: ", len(upsampled_signal))
    print("First 10 samples of upsampled signal: ", upsampled_signal[:10])
    plot_single(upsampled_signal, fs_out, title="Upsampled Signal for Comparison")
