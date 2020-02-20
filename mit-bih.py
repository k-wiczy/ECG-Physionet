##################################################################
# mit-bih.py - skrypt do generowania datasetu mit-bih.           #
# ./data_raw - folder z surowymi danymi MIT-BIH                  #
# mit-bih-ds1.h5 - wyjsciowy plik hdf                            #
# mit-bih-ds1.h5.csv - wyjsciowy plik csv z klasyfikacja danych  #
# Autor : Jakub Wiczynski                                        #
##################################################################

# Import bibliotek i funkcji     
import matplotlib.pyplot as plt
import wfdb
import pandas as pd
import numpy as np
import h5py
from tensorflow.python.autograph.pyct import anno
from qrs_detection import QRSDetectorOffline
from numpy import integer

# Rozgraniczenie bazy na dwa datasety
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 105, 107, 234, 111, 113, 117, 233, 121, 123, 232, 213, 200, 202, 214, 102, 210, 103, 212, 221, 222, 231]

###############################################################################
# Funkcja extract_R - zwraca tablice danych wycentrowanych wokol zespolu QRS
# span             - wielkosc marginesu (T = <center-span ; center+span>
# sampfrom, sampto - granice przeszukiwanego sygnalu
###############################################################################
def extract_R(dataset, span = 100, sampfrom = 0, sampto = None):
    # Get data and classifications from record
    record  = wfdb.rdrecord(('data_raw/' + dataset), sampfrom, sampto)
    R_class = wfdb.rdann(('data_raw/' + dataset), 'atr', sampfrom, sampto)       
    pd.DataFrame(record.p_signal[:, 0]).to_csv("temp.csv")
    R_error = 0
    sampto = len(record.p_signal[:,0])
    
    # Detect QRS
    qrs         = QRSDetectorOffline(ecg_data_path="temp.csv", verbose=True, log_data=False, plot_data=False, show_plot=False)
    peaks       = qrs.detected_peaks_indices
    signal      = np.zeros((span, len(peaks)))
    last_idx    = 0
    
    # Check if all QRS are detected - avoid desynchronization
    if (len(peaks) < len(R_class.symbol)):
        error = 1;
        
    # Move QRS data to array - one element in array contains centered QRS with span defined in arg
    for peak in range(0, len(peaks)):
        if (peaks[peak] + round(span/2) <= (sampto - sampfrom)):
            last_idx = peak
            for i in range(0,span):
                signal[i,peak] = record.p_signal[(peaks[peak] - round(span/2) + i),0]
            
    # Check if the signal has desired length - avoid QRS in start/end of the frame (i.e when QRS is centered at t = 499 and frame has T = 500)
    R_signal = np.zeros((span, last_idx+1))
    for peak in range(0, last_idx+1):
        for i in range(0, span):
            R_signal[i,peak] = signal[i,peak]
        
    # Get classification of each QRS
    R_reference = np.empty((last_idx+1), dtype='object')
    for i in range(0,last_idx+1):
        R_reference[i] = R_class.symbol[i]
    
    # Return classifications and signals
    return (R_error, R_reference, R_signal)  

# Funkcje zlaczania dwoch sygnalow w jeden

def contacenate_signal(in1, in2):
    out_signal = np.zeros((len(in1[:,1]), len(in1[1,:])+len(in2[1,:])))
    for i in range(0, len(in1[1,:])):
        out_signal[:,i] = in1[:,i]
    for i in range(0, len(in2[1,:])):
        out_signal[:,i+len(in1[1,:])] = in2[:,i]
    return out_signal

def contacenate_reference(in1, in2):
    out_reference = np.empty(len(in1)+len(in2), dtype='object')
    for i in range(0, len(in1)):
        out_reference[i] = in1[i]
    for i in range(0, len(in2)):
        out_reference[i+len(in1)] = in2[i]
    return out_reference

# Funkcja extract dataset - wyciaganie kawalkow przebiegu wokol QRS oraz klasyfikacji danego QRSa
def extract_dataset(dataset, span = 100):
    for i in range (0, len(dataset)):
        (error, reference, signal) = extract_R(str(dataset[i]), span)
        if (not error):
            # Copy classification
            if (i == 0):
                dataset_reference = reference;
            else:
                dataset_reference = contacenate_reference(dataset_reference, reference)
            # Copy signal
            if (i == 0):
                dataset_signal = signal;
            else:
                dataset_signal = contacenate_signal(dataset_signal, signal)
    return (dataset_signal, dataset_reference)

# Funkcja get_name - generowanie nazwy datasetu w taki sposob, aby byl w formacie Axxxxxx (6 cyfr)
def get_name(id):
    name = 'A'
    if (id >= 10000):
        name = name + str(id)
    elif (id >= 1000):
        name = name + '0' + str(id)
    elif (id >= 100):
        name = name + '00' + str(id)
    elif (id >= 10):
        name = name + '000' + str(id)
    else:    
        name = name + '0000' + str(id)
    return name

# Funkcja write_hdf - zapis datasetow do pliku hdf
def write_hdf(filename, span = 100):
    (signal, reference) = extract_dataset(DS1, span)
    reference_csv = np.empty((len(signal[0,:]), 2), dtype='object')
    for i in range(0, len(signal[0,:])):
        reference_csv[i,0] = get_name(i)
        reference_csv[i,1] = reference[i]
    np.savetxt(filename + ".csv", reference_csv, delimiter=",", fmt='%s')
    hdf_file = h5py.File(filename, 'w')
    for i in range(0, len(signal[0,:])):  
        print(i)
        group = hdf_file.create_group(get_name(i))
        tsignal = np.zeros((len(signal[:,i]),2))
        for k in range(0,len(signal[:,i])):
            tsignal[k,1] = k / 360
            tsignal[k,0] = signal[k,i]
        ds = group.create_dataset('ecgdata', data=tsignal)
        ds.attrs['baseline']            = [0.0]
        ds.attrs['colnames']            = np.string_('[signal, time_s]')
        ds.attrs['description']         = np.string_('ecgdata')
        ds.attrs['fmt']                 = np.string_('11')
        ds.attrs['gain']                = [1.0]
        ds.attrs['sampling_frequency']  = [360.0]
        ds.attrs['units']               = np.string_('mV')
    hdf_file.close()

write_hdf('mit-bih-ds1.h5', span=500)

print('DONE!')