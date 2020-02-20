import numpy as np
import matplotlib.pyplot as plt
import h5py as h5py
from physionet_processing import special_parameters
from scipy import signal

# get data from hdf file
h5file =  h5py.File("mit-bih-ds1.h5", 'r')
# get a list of dataset names 
dataset_list = list(h5file.keys())
# get parameters 
sequence_lengths, sampling_rates, recording_times, baselines, gains = special_parameters(h5file)
#get min and max values of parameters
sequence_length_min, sequence_length_max = np.min(sequence_lengths), np.max(sequence_lengths)
recording_time_min, recording_time_max = np.min(recording_times), np.max(recording_times)
# based on this, we can set some parameters that we will use in the future
fs = sampling_rates[0] # universal sampling rate
sequence_length = sequence_length_max # will use the maximum sequence length

time = np.arange(0, 100)/fs;
ts = h5file[dataset_list[15]]['ecgdata'][:, 0] # Fetch one time series from the hdf5 file

def interpolate(ts, iterations):
    tsignal = np.zeros(iterations*len(ts))
    offset = 0
    for n in range(0, iterations):
        for k in range(0,len(ts)):
            tsignal[k+(n*len(ts))] = ts[k] + offset
        offset = ts[-1] + offset;
    return tsignal
ts = interpolate(ts, 20)

########################################
#### 1. INTERPOLATE ####################
########################################
fig, ax2 = plt.subplots(figsize = (15, 3))
ax2.plot(ts)
plt.title('Interpolated signal', fontsize = 15)
fig.savefig('physionet_ECG_signal.png', bbox_inches = 'tight', dpi = 150)

########################################
#### 2. FFT ############################
########################################

f1, PSD = signal.periodogram(ts, fs, 'flattop', scaling = 'density')
fig, ax2 = plt.subplots(figsize = (15, 3))
ax2.plot(f1, PSD, 'b')
#ax2.set(xlabel = 'Frequency [Hz]', xlim = [0, time[-1]], xticks = np.arange(0, time[-1]+5, 10))
#ax2.set(ylabel = 'PSD')
plt.title('Power spectral density (PSD)', fontsize = 15)
fig.savefig('physionet_ECG_PSD.png', bbox_inches = 'tight', dpi = 150)



########################################
#### 3. SPECTROGRAM ####################
########################################

from physionet_processing import spectrogram

# Convert ECG into spectrograms without and with log transform
Sx = spectrogram(np.expand_dims(ts, axis = 0), log_spectrogram = False)[2]
Sx_log = spectrogram(np.expand_dims(ts, axis = 0), log_spectrogram = True)[2]

# Get the frequency and time axes
f, t, _ = spectrogram(np.expand_dims(ts, axis = 0), log_spectrogram = False) 

# Plot the spectrograms as images
im_list = [Sx[0], Sx_log[0]]
im_title = ['Spectrogram without log transform', 'Spectrogram with log transform']
fig, ax_list = plt.subplots(1, 2, figsize = (15, 3))

for i, ax in enumerate(ax_list):
    
    ax.imshow(np.transpose(im_list[i]), aspect = 'auto', cmap = 'jet')
    ax.grid(False)
    ax.invert_yaxis()
    ax.set_title(im_title[i], fontsize = 12)
    ax.set(ylim = [0, im_list[i].shape[1]], yticks = np.arange(0, im_list[i].shape[1] + 1, 5))
    ax.set(xlabel = 'Time [s]', ylabel = 'Frequency [Hz]')
    
    # Replace axis labels with time from t array
    xticks_array = np.arange(0, im_list[i].shape[0] + 1, 100)
    ax.set(xlim = [0, im_list[i].shape[0]], xticks = xticks_array)
    labels_new = [str(np.around(t[label], decimals = 1)) for label in xticks_array]
    ax.set_xticklabels(labels_new)
    ax.tick_params(axis = 'x',
                   which = 'both',
                   bottom = 'off')
    
    ax.tick_params(axis = 'y',
                   which = 'both',
                   left = 'off')

plt.tight_layout()
plt.show()
fig.savefig('physionet_ECG_spectrogram.png', bbox_inches = 'tight', dpi = 150)