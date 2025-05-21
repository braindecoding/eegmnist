import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# Bandpass filter dengan scipy (CPU)
def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=128, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Z-score normalization (CPU)
def normalize(x):
    x = np.asarray(x)
    return (x - np.mean(x)) / np.std(x)

# Load Parquet
df = pd.read_parquet("epoc_filtered.parquet")

# Ubah data string ke list float
df["data"] = df["data"].apply(lambda x: list(map(float, x.split(','))))

# Terapkan filter dan normalisasi
df["filtered"] = df["data"].apply(bandpass_filter)
df["normalized"] = df["filtered"].apply(normalize)

# Simpan jika ingin reuse
df = df.drop(columns=["data", "filtered"])
df.to_parquet("epoc_filtered_normalized.parquet")
