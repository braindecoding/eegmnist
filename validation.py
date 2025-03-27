import pandas as pd

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
    
    # Verifikasi apakah jumlah sampel sesuai dengan 'size'
    df_epoc["valid_size"] = df_epoc.apply(lambda row: len(row["data"]) == row["size"], axis=1)
    
    return df_epoc

# Contoh penggunaan
filepath = "EP1.01.txt"  # Ganti dengan path file yang sesuai
df_epoc = load_epoc_data(filepath)

# Menampilkan beberapa baris pertama
print(df_epoc.head())

# Menampilkan ringkasan validasi ukuran sampel
print("Jumlah data yang sesuai dengan 'size':", df_epoc["valid_size"].sum())
print("Jumlah data yang tidak sesuai dengan 'size':", (~df_epoc["valid_size"]).sum())
