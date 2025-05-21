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
    
    return df_epoc

def loadsave69_epoc_data(filepath, output_parquet="epoc_filtered.parquet"):
    columns = ["id", "event", "device", "channel", "code", "size", "data"]
    
    # Baca data
    df = pd.read_csv(filepath, sep='\t', header=None, names=columns)

    # Bersihkan kolom 'device' dan pastikan tipe 'code' numerik
    df["device"] = df["device"].astype(str).str.strip().str.upper()
    df["code"] = pd.to_numeric(df["code"], errors='coerce')  # pastikan numerik

    # Filter: device == 'EP' dan code == 6 atau 9
    df_filtered = df[(df["device"] == "EP") & (df["code"].isin([6, 9]))].copy()

    # Simpan
    df_filtered.to_parquet(output_parquet, index=False)
    print(f"Jumlah data hasil filter: {len(df_filtered)}")
    print(f"Data disimpan ke: {output_parquet}")
    return df_filtered

# Contoh penggunaan
filepath = "EP1.01.txt"  # Ganti dengan path file yang sesuai
df_epoc = loadsave69_epoc_data(filepath)

# Menampilkan beberapa baris pertama
print(df_epoc.head())
