import numpy as np
from scipy.interpolate import lagrange
from scipy import signal
import matplotlib.pyplot as plt
import sample_conversion as sc
import real_time_lagrange as rtla

plt.rcParams['axes.grid'] = True # set all grids to true

wave_duration = 1
sample_rate = 100
f = 2
decimate_factor = 4

samples = wave_duration * sample_rate
print("Number of samples: ", samples)
decimate_samples = int(samples/decimate_factor)
print("Number of decimated samples", decimate_samples)

x = np.linspace(0,wave_duration, samples, endpoint=False)
y = np.sin(x*np.pi*f*2)
# print(type(y))

y_down = sc.decimate(y,decimate_factor)
x_down = np.linspace(0, wave_duration, decimate_samples,endpoint=False)
# print("x downsampled:", x_down)
# print("y downsampled:", y_down)

interpolation_factor = 4
y_padded = sc.zero_padding(y_down,interpolation_factor)
x_padded = np.linspace(0,wave_duration,decimate_samples * interpolation_factor)

# y_interpolated = sc.linear_interpolation(y_padded,interpolation_factor)
y_fir_interpolated = sc.fir_lp_filter(y_padded,cutoff_freq=10)
y_lagrange_interpolated = sc.lagrange_interpolation(y_padded,6)
plt.rcParams['axes.grid'] = True

y_optimal_interpolated = sc.optimal_finite_length_filt(y_padded,6)

sc.lagrange_interpolation_new(y_down,0,4)


# print("y padded: ",y_padded)
# print("x padded: ",x_padded)
# print("Num of samples ypadded",len(y_padded))
# print("Num of samples xpadded",len(x_padded))

fig, axs = plt.subplots(2,3)
fig.suptitle("Upsampling methods")
axs[0,0].plot(x,y,'.-') # original signal
axs[0,0].set_title("Original signal")
axs[0,1].plot(x_down,y_down,'.-') # downsampled signal
axs[0,1].set_title("Downsampled signal")
axs[0,2].plot(x_padded,y_padded,'.-') # zero padded signal
axs[0,2].set_title("Zero padded signal")
axs[1,0].plot(x_padded,y_fir_interpolated,'.-') # linear interpolated signal
axs[1,0].set_title("FIR interpolated signal")
axs[1,1].plot(x_padded,y_lagrange_interpolated,'.-') # lagrange interpolated signal
axs[1,1].set_title("Lagrange interpolated signal")
# axs[1,2].plot(x_padded,y_optimal_interpolated,'.-')
# axs[1,2].set_title("Optimal interpolated signal")
plt.show()
