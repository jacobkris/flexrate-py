import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def mse(signal1, signal2):
    """
    Mean squared error calculation between two signals.
    """
    return np.square(np.subtract(signal1, signal2)).mean()

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

def plot_single(s, fs, title="Signal"):
    """
    Plot one signal
    """
    t = np.arange(0, len(s)) / fs
    plt.plot(t, s, 'b-')
    plt.stem(t, s, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.title(title)
    plt.show()

def plot_double(s1, s2, fs1, fs2, label1 = "Signal 1" , label2 = "Signal 2"):
    """
    Plot two signals with different or same sampling rates
    """
    t1 = np.arange(0, len(s1)) / fs1
    t2 = np.arange(0, len(s2)) / fs2
    plt.figure(figsize=(14, 7))
    plt.stem(t1, s1, linefmt='b-', markerfmt='bo', basefmt='r-', label=label1)
    plt.stem(t2, s2, linefmt='g-', markerfmt='go', basefmt='r-', label=label2)
    plt.title(label1 + " and " + label2)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def fir_precalculate_coeffs(fs_in,fs_out,cutoff,num_taps):
    """
    FIR filter coefficients
    fs_in: input sampling rate
    fs_out: output sampling rate
    cutoff: cutoff frequency
    num_taps: number of taps
    """
    ratio = fs_out / fs_in
    normalized_cutoff = cutoff / fs_in
    coeffs = signal.firwin(num_taps, normalized_cutoff / ratio, window='hamming')
    return coeffs


def lagrange_coeff(t, T, N):
    """
    Lagrange basis polynomial coefficient calculation based on paper formula.  
    t: time instant
    T: sampling interval
    N: interpolation order
    """
    coeff = np.ones(N)
    for i in range(N):
        for l in range(N):
            if i != l:
                coeff[i] *= (t - l * T) / (i * T - l * T)
    return coeff


def fasrc(input_signal, fs_in, fs_out, N, num_taps=None):
    """
    fasrc is an flexible arbitrary ratio resampler that uses either Lagrange interpolation or fir filter for coefficient multiplying.
    input_signal: input signal in int32
    fs_in: input sampling rate
    fs_out: output sampling rate
    N: order of lagrange interpolation
    num_taps: number of taps for fir filter (set to None for lagrange interpolation)
    """

    input_period = 1 / fs 
    output_period = 1 / fs_new 
    output_length = int(np.ceil(len(input_signal) * fs_out / fs_in))
    output_signal = np.zeros(output_length)

    for sample_idx in range(output_length):
        t_k = sample_idx * output_period # current time instant
        n = int(t_k / input_period) # sample index for current time instant
        delta_t = t_k - n * input_period # sampling interval
        offset = 0

        if num_taps is not None:
            coeff = fir_precalculate_coeffs(fs_in, fs_out, 20000, num_taps)
            N = num_taps
            offset = num_taps // 2 # TODO
        else:
            coeff = lagrange_coeff(delta_t, input_period, N) # calculate lagrange coeffs

        # pad input signal if necessary
        if n + N > len(input_signal):
            pad_length = n + N - len(input_signal)
            input_signal = np.pad(input_signal, (0, pad_length), mode='edge')
       
        output_sample = 0
        for i in range(N):
            output_sample += coeff[i] * input_signal[n + i - offset]

        output_signal[sample_idx] = output_sample # append sample to output signal
    
    return np.array(output_signal)

if __name__ == "__main__":
    # plot signal
    f = 440 # 440 Hz
    fs = 41100
    fs_new = 48000
    periods = 8
    duration = periods * 1/f 
    print("Input sampling rate: ", fs)
    print("Output sampling rate: ", fs_new)
    n = int(duration * fs) # number of samples
    t = np.linspace(0, duration, n, endpoint=False)
    n_new = int(duration * fs_new)
    t_new = np.linspace(0, duration, n_new, endpoint=False)

    input_signal = 10000 * np.sin(2 * np.pi * f * t) 
    input_signal = input_signal.astype(np.int32)
    
    real_output_signal = 10000 * np.sin(2 * np.pi * f * t_new)

    output_signal_lagrange = fasrc(input_signal, fs, fs_new, 3)
    output_signal_fir = fasrc(input_signal, fs, fs_new, 3, num_taps=16)

    output_signal_scipy = signal.resample(input_signal, n_new, domain='time')


    plot_double(input_signal, output_signal_lagrange, fs, fs_new, label1="Input Signal", label2="Lagrange Interpolated Output")

    plot_double(input_signal, output_signal_fir, fs, fs_new, label1="Input Signal", label2="FIR Interpolated Output")

    plot_double(output_signal_lagrange, real_output_signal, fs_new, fs_new, label1="Lagrange Interpolated Output", label2="Real Interpolated")
    
    plot_double(output_signal_scipy, real_output_signal, fs_new, fs_new, label1="Scipy Interpolated Output", label2="Real Interpolated")

    # Mean squared error
    mse_scipy = mse(output_signal_scipy, real_output_signal)
    mse = mse(output_signal_lagrange, real_output_signal)
    print("Mean squared error between real and my estimated: ", mse)
    print("Mean squared error between real and scipy: ", mse_scipy)

    # test mse of different interpolation orders
