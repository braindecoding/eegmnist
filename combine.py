import pandas as pd
import numpy as np

# Load data dari Parquet
df = pd.read_parquet("epoc_filtered_normalized.parquet")

# Konversi data string → list of float (kalau masih string)
df["normalized"] = df["normalized"].apply(lambda x: list(map(float, x.split(','))) if isinstance(x, str) else x)

# Fungsi untuk menggabungkan semua channel EEG dalam 1 event jadi 1 vektor
def combine_channels(group):
    vectors = group["normalized"].tolist()  # List of lists (or arrays)
    combined = np.concatenate(vectors)
    return pd.Series({"eeg_vector": combined})

# Gabungkan berdasarkan event
combined_data = df.groupby("event").apply(combine_channels).reset_index()

# Ambil info code per event
df_code = df[["event", "code"]].drop_duplicates()

# Gabungkan info code
result = pd.merge(combined_data, df_code, on="event")

# Konversi eeg_vector dari NumPy array ke list agar bisa disimpan ke Parquet
result["eeg_vector"] = result["eeg_vector"].apply(lambda x: x.tolist())

# Simpan ke Parquet
result.to_parquet("eeg_combined_by_event.parquet", index=False)

# Tampilkan beberapa baris hasil
print(result.head())
