import cudf
import cupy as cp
import ast

# Load data dari parquet
df = cudf.read_parquet("epoc_filtered.parquet")

# Konversi data string → list of float
df["data"] = df["data"].applymap(lambda x: cp.array(list(map(float, x.split(',')))))

# Gabungkan data dari semua channel untuk setiap event
grouped = df.groupby("event")

# Fungsi untuk menggabungkan channel EEG menjadi satu vektor
def combine_channels(sub_df):
    vectors = sub_df["data"].to_arrow().to_pylist()
    combined = cp.concatenate(vectors)
    return combined

# Buat DataFrame hasil akhir
combined_data = grouped.apply_grouped(
    combine_channels,
    incols=["data"],
    outcols={"eeg_vector": cp.ndarray},
    apply_chunksize=1
)

# Gabungkan kembali info code per event (karena semua code sama dalam 1 event)
df_code = df[["event", "code"]].drop_duplicates()
result = combined_data.merge(df_code, on="event")
result.to_parquet("event_combined", index=False)
print(result.head())
