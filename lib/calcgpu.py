# Versi GPU-ready untuk fungsi S() menggunakan FAISS (GPU) dan CuPy
import numpy as np
import cupy as cp
import faiss
from time import time

def S_gpu(k, t, Y_train, Y_test):
    """
    Hitung matriks S menggunakan FAISS (GPU) dan CuPy.
    """
    start_time = time()

    # Convert to float32 (required by FAISS)
    Y_train = Y_train.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    # Dimensi data
    ntrn = Y_train.shape[0]
    ntest = Y_test.shape[0]

    # Buat FAISS GPU index
    d = Y_train.shape[1]
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)

    # Tambahkan data training ke index
    gpu_index.add(Y_train)

    # Cari k+1 tetangga (karena yang pertama adalah dirinya sendiri)
    distances, indices = gpu_index.search(Y_test, k + 1)

    # Buat matriks S (CuPy untuk GPU ops, lalu ke NumPy)
    S = cp.zeros((ntrn, ntest), dtype=cp.float32)
    for i in range(ntest):
        # Ambil top-k tetangga (tanpa dirinya sendiri)
        idx = indices[i][1:]  # indeks di Y_train
        dist = distances[i][1:]  # jaraknya
        weights = cp.exp(-cp.asarray(dist) / (2 * (t ** 2)))  # Heat kernel
        S[idx, i] = weights

    S = cp.asnumpy(S)  # Kembalikan ke NumPy jika mau disimpan atau diproses lanjut

    print("GPU-based S() selesai dalam %.2f detik." % (time() - start_time))
    return S
