import pandas as pd
import numpy as np

# Load data
df = pd.read_parquet("epoc_filtered_normalized.parquet")

# Pastikan kolom 'normalized' berupa list of float
df["normalized"] = df["normalized"].apply(lambda x: list(map(float, x.split(','))) if isinstance(x, str) else x)

# Fungsi cek ukuran data per event berdasarkan kolom normalized
def check_size(group):
    sizes = [len(x) for x in group["normalized"]]
    return pd.Series({
        "num_channels": len(sizes),
        "all_same_length": len(set(sizes)) == 1,
        "length_per_channel": sizes[0] if sizes else 0
    })

size_check = df.groupby("event").apply(check_size).reset_index()
print("Ukuran tiap event:\n", size_check)

# Cek event yang bermasalah (channel panjangnya beda atau jumlah channel beda)
bad_events = size_check[(size_check["all_same_length"] == False) | (size_check["num_channels"] != size_check["num_channels"].iloc[0])]
print("Event dengan ukuran channel tidak konsisten:\n", bad_events)

# Hapus event bermasalah (opsional, bisa sesuaikan)
if len(bad_events) > 0:
    df = df[~df["event"].isin(bad_events["event"])]

# Gabung channel jadi vector EEG per event
def combine_channels(group):
    vectors = group["normalized"].tolist()
    combined = np.concatenate(vectors)
    return combined

combined_data = df.groupby("event")["normalized"].apply(lambda g: np.concatenate(g.tolist())).reset_index()
combined_data = combined_data.rename(columns={"normalized": "eeg_vector"})

# Fix panjang vector EEG per event jadi 256
fixed_length = 256

def pad_or_truncate(vec, fixed_length):
    vec = np.array(vec)
    if len(vec) > fixed_length:
        return vec[:fixed_length]
    elif len(vec) < fixed_length:
        pad_width = fixed_length - len(vec)
        return np.pad(vec, (0, pad_width), 'constant')
    else:
        return vec

combined_data["eeg_vector"] = combined_data["eeg_vector"].apply(lambda x: pad_or_truncate(x, fixed_length).tolist())

# Ambil kode per event
df_code = df[["event", "code"]].drop_duplicates()
result = pd.merge(combined_data, df_code, on="event")

# Simpan ke parquet
result.to_parquet("eeg_combined_by_event_fixed_length.parquet", index=False)

print(result.head())
