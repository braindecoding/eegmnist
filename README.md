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
