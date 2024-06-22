import numpy as np
from scipy import signal

def decimate(x,decimation_factor):
    y = signal.decimate(x,decimation_factor)
    return y

def zero_padding(x,interpolation_factor):
    y_tmp = np.zeros(len(x)*interpolation_factor)

    for i in range(0,len(x)):
        y_tmp[i * interpolation_factor] = x[i]

    y = y_tmp 
    return y

def linear_interpolation(x,interpolation_factor):
    # y = [value1,0,0,0,value2,0,0,0,...]
    y_tmp = []
    for i in range(0,len(x),interpolation_factor):
        y_tmp.append(x[i])
        for j in range(1,interpolation_factor):
            y_tmp.append(x[i + j])
    y = y_tmp
    return y 

def fir_lp_filter(x, cutoff_freq):
    # for loop?
    num_taps = 8
    nyquist_rate = 0.5 * 100 * 4
    normalized_cutoff = cutoff_freq / nyquist_rate
    taps = signal.firwin(num_taps,normalized_cutoff,window='hamming')
    y = signal.filtfilt(taps,1.0,x)
    
    return y

def lagrange_interpolation(x,order):
    N = len(x)
    x_interp = np.zeros(N)
    
    for n in range(N):
        indices = np.arange(max(0, n - order), min(N, n + order + 1)) # nearby points
        indices = indices[x[indices] != 0]  # take out zero
        if len(indices) > 1: 
            t = indices
            y = x[indices]
            
            # calculate lagrange and sum for each
            lagrange_basis = np.array([np.prod([(n - t[m]) / (t[j] - t[m]) for m in range(len(t)) if m != j]) for j in range(len(t))])
            x_interp[n] = np.sum(y * lagrange_basis)
        else:
            x_interp[n] = x[n]  # copy if cannot interpolate
    
    return x_interp
