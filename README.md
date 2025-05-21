# eegmnist
Braindecoding on MNIST Stimuli with EEG as input signal
1. Using [MindBigData](https://mindbigdata.com/opendb/index.html) dataset
2. Focusing in [Emotiv EPOC device](https://www.emotiv.com/products/epoc-x)
3. Link for [dataset](https://mindbigdata.com/opendb/MindBigData-EP-v1.0.zip)


## Validasi dataset

![image](https://github.com/user-attachments/assets/fde68e6e-4001-4321-b7c2-b9f8f022cc63)

Kolom **size** dalam dataset menunjukkan jumlah sampel yang dikumpulkan selama 2 detik oleh perangkat EEG.  

Karena perangkat **EPOC** memiliki frekuensi sampling sekitar **128 Hz**, maka dalam 2 detik seharusnya ada sekitar:  

128 x 2 = 256 

Namun, dalam contoh yang kamu tampilkan, nilai **size** adalah **260**, yang sedikit lebih banyak dari 256. Ini bisa disebabkan oleh:
1. Variasi dalam proses rekaman EEG yang menyebabkan perbedaan jumlah sampel.
2. Overlap kecil dalam rekaman untuk memastikan data tidak terputus.
3. Pengolahan sinyal tambahan yang dilakukan oleh perangkat sebelum disimpan.

untuk memverifikasi jumlah sampel dalam kolom `data` sesuai dengan nilai `size` jalankan file `validation.py`

![image](https://github.com/user-attachments/assets/46438b9c-e5a6-425e-8b06-f41e22d2ae11)

## Langkah

Untuk melakukan *brain decoding*:  

### **Preprocessing**  

- Normalisasi sinyal EEG  
- Filtering (bandpass 0.5–50 Hz)  
- Epoching jika diperlukan  


1. **Normalisasi**  
   - EEG memiliki rentang nilai yang bervariasi antar individu dan sesi rekaman.  
   - Normalisasi membantu model belajar pola yang lebih stabil dengan menyamakan skala data.  
   - Metode yang umum digunakan:
     - **Z-score normalization**: \( X' = \frac{X - \mu}{\sigma} \) (menghilangkan mean dan membagi dengan standar deviasi).  
     - **Min-max scaling**: \( X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}} \) (memetakan ke rentang [0,1] atau [-1,1]).  

2. **Epoching**  
   - Epoching memotong sinyal EEG menjadi segmen-segmen kecil berdasarkan event tertentu.  
   - Biasanya digunakan dalam analisis **Event-Related Potential (ERP)** dan *brain decoding* berbasis stimulus.  
   - Untuk dataset kamu, bisa dilakukan **epoching per event atau per code**.  




### **Feature Extraction**  
- Power Spectral Density (PSD)  
- Wavelet Transform  
- Statistical features (mean, variance, entropy)  

Ya, **feature extraction sangat penting** dalam *brain decoding* karena sinyal EEG memiliki dimensi tinggi dan banyak variabilitas. Dengan ekstraksi fitur, kita bisa mendapatkan representasi yang lebih kompak dan bermakna untuk analisis atau klasifikasi.**Fitur yang Umum Digunakan untuk EEG**

1. **Fitur Waktu (Time-Domain Features)**
   - **Mean**: Rata-rata nilai sinyal dalam epoch.
   - **Standard Deviation**: Variabilitas sinyal.
   - **Root Mean Square (RMS)**: Kekuatan sinyal dalam epoch.
   - **Skewness & Kurtosis**: Karakteristik distribusi sinyal.

2. **Fitur Frekuensi (Frequency-Domain Features)**
   - **Power Spectral Density (PSD)**: Kekuatan sinyal pada berbagai frekuensi.
   - **Band Power**: Energi sinyal dalam pita frekuensi tertentu:
     - Delta (0.5–4 Hz) → Tidur nyenyak
     - Theta (4–8 Hz) → Relaksasi
     - Alpha (8–13 Hz) → Ketenangan
     - Beta (13–30 Hz) → Fokus/konsentrasi
     - Gamma (>30 Hz) → Pemrosesan tinggi

3. **Fitur Waktu-Frekuensi (Time-Frequency Features)**
   - **Wavelet Transform**: Mengekstrak pola dalam domain waktu dan frekuensi.



3. **Classification Model**  
   - CNN/LSTM untuk deep learning  
   - SVM/Random Forest untuk metode konvensional  



## Inovasi

Model:
1. Generative Adversarial Network
2. Diffusion Model
