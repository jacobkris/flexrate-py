import numpy as np

def fasrc(input_signal, input_rate, output_rate, farrow_coeffs):
    """
    Fractional sampling rate conversion using farrow structure.
    input_signal: input signal
    input_rate: input rate
    output_rate: output rate (wanted rate)
    farrow_coeffs: Farrow coefficients (matrix of shape (poly_order + 1, num_subfilt)
    """
    poly_order = farrow_coeffs.shape[0] - 1
    num_subfilt = farrow_coeffs.shape[1]
    padded_input = np.pad(input_signal, (num_subfilt - 1, 0), mode='constant')
    ratio = output_rate / input_rate

    output_length = int(np.ceil(len(input_signal) * ratio))
    output_signal = np.zeros(output_length)

    for sample_idx in range(output_length):
        frac_delay = sample_idx / ratio 
        int_delay = int(frac_delay)
        mu = frac_delay - int_delay 
        
        for i in range(num_subfilt):
            x = padded_input[int_delay + num_subfilt - 1 - i] # adds padding to edges
            poly_value = 0
            for j in range(poly_order + 1):
                poly_value += farrow_coeffs[j][i] * (mu ** j)
            output_signal[sample_idx] += x * poly_value

    return output_signal

