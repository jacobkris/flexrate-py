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

def design_farrow_coeffs(poly_order, num_subfilters , fc=0.4):
    """
    Farrow filter coefficients calculation
    poly_order: polynomial order
    num_subfilt: number of subfilters
    """
    coeffs = [
        0.0134723321567464, 0.0303640851789094, -0.00727985338412466, -0.0171107153161435, 0.0346995556841227, -0.0235535999197771, -0.0238593113883345, 0.0803821737365246, -0.0877592868647261, -0.0293147070318769, 0.564499995759284, 0.564499995759284, -0.0293147070318769, -0.0877592868647261, 0.0803821737365246, -0.0238593113883345, -0.0235535999197771, 0.0346995556841227, -0.0171107153161435, -0.00727985338412466, 0.0303640851789094, 0.0134723321567464
    ]
    p = np.polynomial.polynomial.Polynomial.fit(np.arange(0, len(coeffs)), coeffs, poly_order)



def fasrc(input_signal, input_rate, output_rate, farrow_coeffs=None):
    # initialize coefficients
    if farrow_coeffs is None:
        farrow_coeffs = np.array([
            [-8.57738278e-3, 7.82989032e-1, 7.19303539e+0, 6.90955718e+0, -2.62377450e+0, -6.85327127e-1, 1.44681608e+0, -8.79147907e-1, 7.82633997e-2, 1.91318985e-1, -1.88573400e-1, 6.91790782e-2, 3.07723786e-3, -6.74800912e-3],
            [2.32448021e-1, 2.52624309e+0, 7.67543936e+0, -8.83951796e+0, -5.49838636e+0, 6.07298348e+0, -2.16053205e+0, -7.59142947e-1, 1.41269409e+0, -8.17735712e-1, 1.98119464e-1, 9.15904145e-2, -9.18092030e-2, 2.74136108e-2],
            [-1.14183319e+0, 6.86126458e+0, -6.86015957e+0, -6.35135894e+0, 1.10745051e+1, -3.34847578e+0, -2.22405694e+0, 3.14374725e+0, -1.68249886e+0, 2.54083065e-1, 3.22275037e-1, -3.04794927e-1, 1.29393976e-1, -3.32026332e-2],
            [1.67363115e+0, -2.93090391e+0, -1.13549165e+0, 5.65274939e+0, -3.60291782e+0, -6.20715544e-1, 2.06619782e+0, -1.42159644e+0, 3.75075865e-1, 1.88433333e-1, -2.64135123e-1, 1.47117661e-1, -4.71871047e-2, 1.24921920e-2]
        ]) / 12.28

    poly_order = farrow_coeffs.shape[0] - 1
    num_subfilt = farrow_coeffs.shape[1]

    ratio = output_rate / input_rate

    output_length = int(np.ceil(len(input_signal) * ratio))
    output_signal = np.zeros(output_length)

    for sample_idx in range(output_length):
        frac_delay = sample_idx / ratio
        int_delay = int(frac_delay)
        mu = frac_delay - int_delay
        
        for i in range(num_subfilt):
            x = input_signal[int_delay - i] if int_delay - i >= 0 else 0
            poly_value = 0
            for j in range(poly_order + 1):
                poly_value += farrow_coeffs[j][i] * (mu ** j)
            output_signal[sample_idx] += x * poly_value

    return output_signal

if __name__ == "__main__":
    
    f = 6000 # 440 Hz
    fs = 44100
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
    
    output_signal_ideal = 10000 * np.sin(2 * np.pi * f * t_new)

    output_signal_scipy = signal.resample(input_signal, n_new, domain='time')

    output_signal_fasrc = fasrc(input_signal, fs, fs_new)
    print("Input signal length: ", len(input_signal))
    print("Output signal length: ", len(output_signal_ideal))
    print("Output Farrow signal length: ", len(output_signal_fasrc))

    plot_double(input_signal, output_signal_fasrc, fs, fs_new, label1="Input Signal", label2="Farrow Interpolated Output")
    plot_double(output_signal_ideal, output_signal_fasrc, fs_new, fs_new, label1="Ideal Output", label2="Farrow Interpolated Output")
    plot_double(output_signal_ideal, output_signal_scipy, fs_new, fs_new, label1="Scipy Output", label2="Farrow Interpolated Output")

    # Mean squared error
    mse_fasrc = mse(output_signal_fasrc, output_signal_ideal)
    mse_scipy = mse(output_signal_scipy, output_signal_ideal)
    print("Mean squared error between real and my estimated: ", mse_fasrc)
    print("Mean squared error between real and scipy: ", mse_scipy)

    # Farrow coefficients
    farrow_coeffs = design_farrow_coeffs(3, 4)
    print(farrow_coeffs)

