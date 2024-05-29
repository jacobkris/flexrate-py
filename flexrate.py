import numpy as np
import matplotlib.pyplot as plt


def lagrange_coeff(t_d,x,k,N):
    coeffs = np.zeros(N)
    for k in range(N):
        product = 1
        for j in range(N):
            if j != k:
                product *= (t_d - x[j])/(x[k] - x[j])
        coeffs[k] = product
    return coeffs

def flexrate():
   return None

if __name__ == "__main__":
    duration = 1.0
    input_rate = 41100
    output_rate = 48000
    t = np.linspace(0,duration,int(duration*input_rate),endpoint=False)
    input_signal = np.sin(2 * np.pi * 440 * t) # 440 Hz sine wave
    input_signal = input_signal.astype(np.int32)
    print(input_signal)

