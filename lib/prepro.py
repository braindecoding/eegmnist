from scipy.io import loadmat

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from skimage.transform import resize

def getXY(filepath,resolution):
    handwriten_69=loadmat(filepath)
    #ini fmri 10 test 90 train satu baris berisi 3092
    Y_train = handwriten_69['fmriTrn'].astype('float32')
    Y_test = handwriten_69['fmriTest'].astype('float32')
    
    # ini stimulus semua
    X_train = handwriten_69['stimTrn']#90 gambar dalam baris isi per baris 784 kolom
    X_test = handwriten_69['stimTest']#10 gambar dalam baris isi 784 kolom
    
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    
    #channel di depan
    #X_train = X_train.reshape([X_train.shape[0], 1, resolution, resolution])
    #X_test = X_test.reshape([X_test.shape[0], 1, resolution, resolution])
    #channel di belakang(edit rolly) 1 artinya grayscale
    X_train = X_train.reshape([X_train.shape[0], resolution, resolution, 1])
    X_test = X_test.reshape([X_test.shape[0], resolution, resolution, 1])
    # In[]: Normlization sinyal fMRI, min max agar nilainya hanya antara 0 sd 1
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))   
    Y_train = min_max_scaler.fit_transform(Y_train)     
    Y_test = min_max_scaler.transform(Y_test)
    return X_train,X_test,Y_train,Y_test


def getXYVal(filepath, resolution, validation_size=0.2, random_state=None):
    # Load data from .mat file
    handwriten_69 = loadmat(filepath)
    
    # Extract fMRI signals for training and testing
    Y_train = handwriten_69['fmriTrn'].astype('float32')
    Y_test = handwriten_69['fmriTest'].astype('float32')
    
    # Extract stimulus images for training and testing
    X_train = handwriten_69['stimTrn'].astype('float32') / 255.
    X_test = handwriten_69['stimTest'].astype('float32') / 255.
    
    # Reshape the input images to have the channel in the last dimension (for convolutional networks)
    X_train = X_train.reshape([X_train.shape[0], resolution, resolution, 1])
    X_test = X_test.reshape([X_test.shape[0], resolution, resolution, 1])
    
    # Normalize fMRI signals to have values between 0 and 1
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    Y_train = min_max_scaler.fit_transform(Y_train)
    Y_test = min_max_scaler.transform(Y_test)
    
    # Split the data into training and validation sets
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train, test_size=validation_size, random_state=random_state
    )
    
    return X_train, X_test, X_validation, Y_train, Y_test, Y_validation

def getXYValeeg(filename, resolution=28, test_size=0.2, val_size=0.1, random_state=42):
    # Load MNIST dari sklearn (fetch_openml)
    mnist = fetch_openml('mnist_784', version=1)
    X_mnist = mnist.data.values.reshape(-1, 28, 28).astype(np.uint8)
    y_mnist = mnist.target.astype(int).values

    # Load data
    df = pd.read_parquet(filename)

    # Pastikan eeg_vector jadi numpy array
    df["eeg_vector"] = df["eeg_vector"].apply(np.array)

    # Siapkan dictionary label -> list gambar MNIST sesuai label
    mnist_dict = {}
    for digit in range(10):
        mnist_dict[digit] = X_mnist[y_mnist == digit]

    # Fungsi ubah code ke gambar MNIST (ambil gambar pertama dari mnist_dict[label])
    def code_to_image(code):
        imgs = mnist_dict.get(code)
        if imgs is None or len(imgs) == 0:
            return np.zeros((resolution, resolution), dtype=np.float32)
        img = imgs[0]
        if img.shape != (resolution, resolution):
            img = resize(img, (resolution, resolution), anti_aliasing=True)
        img = img.astype(np.float32) / 255.0
        return img

    # Buat kolom gambar X dari code
    df["X_img"] = df["code"].apply(code_to_image)

    # X: gambar, Y: eeg_vector
    X = np.stack(df["X_img"].values)  # shape (n_samples, res, res)
    Y = np.stack(df["eeg_vector"].values)  # shape (n_samples, fixed_length)

    # Split train + temp_test
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=(test_size + val_size), random_state=random_state, stratify=df["code"]
    )

    # Hitung proporsi val dari X_temp
    val_ratio = val_size / (test_size + val_size)

    # Split test dan val dari X_temp
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=val_ratio, random_state=random_state, stratify=df.loc[df.index.isin(df.index.difference(X_train.shape[0])), "code"] if len(df) > 0 else None
    )

    # Reshape channel terakhir untuk CNN (grayscale)
    X_train = X_train.reshape([X_train.shape[0], resolution, resolution, 1])
    X_test = X_test.reshape([X_test.shape[0], resolution, resolution, 1])
    X_val = X_val.reshape([X_val.shape[0], resolution, resolution, 1])

    return X_train, X_test, X_val, Y_train, Y_test, Y_val
