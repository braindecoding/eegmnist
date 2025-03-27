import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# Fungsi untuk menerapkan bandpass filter
def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=128, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Fungsi untuk normalisasi data menggunakan Z-score
def normalize_data(data):
    data = np.array(data)
    return (data - np.mean(data)) / np.std(data)

# Fungsi untuk membaca dataset dan memfilter hanya data dari perangkat EPOC
def load_epoc_data(filepath):
    # Kolom dalam dataset
    columns = ["id", "event", "device", "channel", "code", "size", "data"]
    
    # Membaca file dengan pemisah tab tanpa header
    df = pd.read_csv(filepath, sep='\t', header=None, names=columns, dtype={"device": str})
    
    # Filter hanya untuk perangkat EPOC
    df_epoc = df[df["device"] == "EP"].copy()
    
    # Mengonversi kolom 'data' dari string menjadi daftar angka
    df_epoc["data"] = df_epoc["data"].apply(lambda x: list(map(float, x.split(','))))
    
    # Terapkan bandpass filter
    df_epoc["filtered_data"] = df_epoc["data"].apply(lambda x: bandpass_filter(np.array(x)))
    
    # Terapkan normalisasi Z-score
    df_epoc["normalized_data"] = df_epoc["filtered_data"].apply(lambda x: normalize_data(x))
    
    return df_epoc

# Contoh penggunaan
filepath = "EP1.01.txt"  # Ganti dengan path file yang sesuai
df_epoc = load_epoc_data(filepath)

# Menampilkan beberapa baris pertama
print(df_epoc.head())
