####################################################################################
# evaluate.py - Skrypt do testowania modelu po uczeniu.                            #
# mit-bih-ds1.h5 - Dataset danych testujacych (MIT-BIH)                            #    
# mit-bih-ds1.h5.csv - plik csv z klasyfikacja danych                              #
# model_augmented.h5 - dane modelu, ktory jest testowany                           #
# Autor : Jakub Wiczynski - modyfikacja kodu physionet ( Andreas Werdich)          #
# Modyfikacja polegala na dostosowaniu skryptu do dzialania na krotszych           #
# przebiegach, skupionych wokol zespolu QRS z zadanym marginesem i dostosowaniem   #
# skryptu do datasetu MIT-BIH                                                      #
####################################################################################   

# Import bibliotek i funkcji                     
import numpy as np
import h5py as h5py
import pandas as pd
import tensorflow as tf
import keras
from physionet_processing import special_parameters
from physionet_processing import spectrogram
from sklearn.preprocessing import LabelEncoder
from physionet_generator import DataGenerator


###########################################################
# Pobranie danych testujacych i klasyfikatorow
###########################################################

# Wczytanie danych datasetu z pliku hdf
h5file =  h5py.File("mit-bih-ds1.h5", 'r')
dataset_list = list(h5file.keys())

# Parametry datasetow
sequence_lengths, sampling_rates, recording_times, baselines, gains = special_parameters(h5file)

# Minimalne i maksymalne parametry
sequence_length_min, sequence_length_max = np.min(sequence_lengths), np.max(sequence_lengths)
recording_time_min, recording_time_max = np.min(recording_times), np.max(recording_times)

# Czestotliwosc probkowania, dlugosc przebiegu czasowego, skala czasu
fs = sampling_rates[0]
sequence_length = sequence_length_max
ts = h5file[list(h5file.keys())[20]]['ecgdata'][:, 0]
time = np.arange(0, len(ts))/fs

# Wczytanie klasyfikacji poszczegolnych przebiegow
label_df = pd.read_csv("mit-bih-ds1.h5.csv", header = None, names = ['name', 'label'])
label_set = list(sorted(label_df.label.unique()))
encoder = LabelEncoder().fit(label_set)
label_set_codings = encoder.transform(label_set)
label_df = label_df.assign(encoded = encoder.transform(label_df.label))

# Wyswietlenie klasyfikatorow
print(label_set)

###########################################################
# Przypisanie poszczegolnych przebiegow do poszczegolnej klasyfikacji
###########################################################

label_list = list()
for i in np.arange(0, label_df.shape[0]):
    if label_df.iloc[i].label == 'N':
        label_list.append(label_df.iloc[i].name)
    
partition = { 'N' : list(label_df.iloc[label_list,].name) }
labels_dict = dict(zip(label_df.name, label_df.encoded))

###########################################################
# Generowanie danych testujacych model - klasa DataGenerator
###########################################################

# Maksymalna dlugosc przebiegu
max_length = sequence_length

# Parametry spektrogramu
sequence_length = sequence_length_max 
spectrogram_nperseg = 64
spectrogram_noverlap = 32
n_classes = len(label_df.label.unique())
batch_size = 32

# Wymiary obrazu okreslone za pomoca wymiarow spektrogramu
Sx_log = spectrogram(np.expand_dims(ts, axis = 0),
                     nperseg = spectrogram_nperseg,
                     noverlap = spectrogram_noverlap,
                     log_spectrogram = True)[2]
dim = Sx_log[0].shape

# Inicjalizacja parametrow klasy DataGenerator

params = {'batch_size': batch_size,
          'dim': dim,
          'nperseg': spectrogram_nperseg,
          'noverlap': spectrogram_noverlap,
          'n_channels': 1,
          'sequence_length': sequence_length,
          'n_classes': n_classes,
          'shuffle': True}


##############################################################
# Testowanie danych o poszczegolnych klasyfikacjach na modelu
##############################################################

for label_idx in np.arange(0, 2):
    val_generator = DataGenerator(h5file, partition['N'], labels_dict, augment = False, **params)
    for i, batch in enumerate(val_generator):
        if i == 1:
            break
    X = batch[0]
    y = batch[1]
    model = keras.models.load_model('model_augmented.h5', custom_objects={'tf': tf})
    score = model.evaluate(X,y,batch_size = 32)
    print(score)

    
    
    