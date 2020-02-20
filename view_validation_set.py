import numpy as np
import matplotlib.pyplot as plt
import h5py as h5py
import pandas as pd
from physionet_processing import fetch_h5data

np.random.seed(10)

df = pd.read_csv("mit-bih-ds2.h5.csv", header = None, names = ['name', 'label'])
df_stat = df.groupby('label').agg('count').reset_index()
df_stat.columns = ['label', 'recordings']
df_stat = df_stat.assign(percent = (100 * np.around(df_stat.recordings/df.shape[0], 2)).astype(np.int))
df_set = list(df.label.unique())

h5file =  h5py.File("mit-bih-ds2.h5", 'r')
dataset_list = list(h5file.keys())

def get_sample():
    # Pick one ECG randomly from each class 
    fid_list = [np.random.choice(df[df.label == label].name.values, 1)[0] for label in df_set]
    return fid_list

name_list = get_sample()

print(df_stat)
slen = 100

idx_list = [dataset_list.index(name) for name in name_list]
data = fetch_h5data(h5file, idx_list, sequence_length = slen)
time = np.arange(0, slen)/300

fig, ax = plt.subplots(3, 5, figsize=(20,5))
for row in np.arange(0,3):
    for i, ax1 in enumerate(ax[row,:]):
        ax1.plot(time, data[i+(row*5)], color = 'r')
        ax1.set_title(name_list[i+(row*5)] + ' class:' + df_set[i+(row*5)])

plt.tight_layout()
plt.show()    
fig.savefig('view_validation_set.png', bbox_inches = 'tight', dpi = 150)