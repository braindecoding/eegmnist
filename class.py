import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

# Fungsi untuk ekstraksi fitur waktu
def extract_time_features(data):
    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "rms": np.sqrt(np.mean(np.square(data))),
        "skewness": pd.Series(data).skew(),
        "kurtosis": pd.Series(data).kurtosis()
    }

# Fungsi untuk ekstraksi fitur frekuensi (Power Spectral Density dan Band Power)
def extract_frequency_features(data, fs=128):
    freqs, psd = welch(data, fs, nperseg=len(data))
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
    }
    band_power = {}
    for band, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power[f"power_{band}"] = np.sum(psd[idx])
    return band_power

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
    
    # Ekstraksi fitur waktu dan frekuensi
    time_features = df_epoc["normalized_data"].apply(lambda x: extract_time_features(x))
    freq_features = df_epoc["normalized_data"].apply(lambda x: extract_frequency_features(x))
    
    # Menggabungkan fitur ke dataframe
    df_features = pd.concat([df_epoc, time_features.apply(pd.Series), freq_features.apply(pd.Series)], axis=1)
    
    return df_features

# Contoh penggunaan
filepath = "EP1.01.txt"  # Ganti dengan path file yang sesuai
df_epoc = load_epoc_data(filepath)

# Menyiapkan data untuk klasifikasi
feature_cols = ["mean", "std", "rms", "skewness", "kurtosis", "power_delta", "power_theta", "power_alpha", "power_beta", "power_gamma"]
X = df_epoc[feature_cols]
y = df_epoc["code"]  # Label target (digit yang dipikirkan/dilihat)

# Membagi data menjadi training dan testing (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model SVM
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Memprediksi data uji
y_pred = clf.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy * 100:.2f}%")
