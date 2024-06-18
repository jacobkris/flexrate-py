from types import new_class
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def plot_signals(input_signal, output_signal, fs, fs_new):
    t_input = np.arange(0, len(input_signal)) / fs
    t_output = np.arange(0, len(output_signal)) / fs_new
    
    plt.figure(figsize=(14, 7))
    
    plt.stem(t_input, input_signal, linefmt='b-', markerfmt='bo', basefmt='r-', label='Input Signal')
    # plt.plot(t_input, input_signal, 'b--', label='Input Signal')
    
    plt.stem(t_output, output_signal, linefmt='g-', markerfmt='go', basefmt='r-', label='Output Signal')
    # plt.plot(t_output, output_signal, 'g--', label='Output Signal')
    
    plt.title('Input and Upsampled Output Signals')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.show()


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

def fir_lowpass_coeff(fc, fs, N):


def flexrate(input_signal, fs, fs_new, N, lowpass=None):
    """
    Flexrate is an arbitrary ratio resampler that uses Lagrange interpolation.
    signal: input signal in int32
    fs: input sampling rate
    fs_new: output sampling rate
    N: order of lagrange interpolation
    """
    if lowpass is not None:
        coeff = np.convolve(coeff, lowpass, mode='same')

    T = 1 / fs 
    T_new = 1 / fs_new 
    output_signal = []
    for t_new in np.arange(0, len(input_signal) * T, T_new): 
        n = int(t_new / T) # sample index for current time instant
        t_d = t_new - n * T # sampling interval
        coeff = lagrange_coeff(t_d, T, N) # calculate lagrange coeffs
        

        
        # pad input signal if necessary
        if n + N > len(input_signal):
            pad_length = n + N - len(input_signal)
            input_signal = np.pad(input_signal, (0, pad_length), mode='edge')
        
        output_sample = np.dot(coeff, input_signal[n:n+N]) # calculate output sample with dot product
        output_signal.append(output_sample) # append sample to output signal
    
    return np.array(output_signal)

if __name__ == "__main__":
    # plot signal
    f = 440 # 440 Hz
    fs = 41100
    fs_new = 48000
    periods = 4
    duration = periods * 1/f 
    print("Input sampling rate: ", fs)
    print("Output sampling rate: ", fs_new)
    n = int(duration * fs) # number of samples
    t = np.linspace(0, duration, n, endpoint=False)
    n_new = int(duration * fs_new)
    t_new = np.linspace(0, duration, n_new, endpoint=False)

    input_signal = 10000 * np.sin(2 * np.pi * f * t) 
    input_signal = input_signal.astype(np.int32)
    print("Input signal: ", input_signal[:10])

    output_signal = flexrate(input_signal, fs, fs_new, 3)
    print("Output signal length: ", len(output_signal))
    print("Output signal samples: ", output_signal)
    print("type of output signal: ", type(output_signal))

    output_signal_scipy = signal.resample(input_signal, n_new,domain='time')
    print("Output signal length (scipy): ", len(output_signal_scipy))
    print("Output signal samples (scipy): ", output_signal_scipy)
    print("type of output signal (scipy): ", type(output_signal_scipy))

    plot_signals(input_signal, output_signal, fs, fs_new)

    # test resampling with scipy

    # error difference from original signal

    plot_signals(input_signal, output_signal_scipy, fs, fs_new)

