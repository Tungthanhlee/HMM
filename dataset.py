import librosa
import numpy as np
import os
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
from config import DATA_PATH, CLASS_NAMES


def get_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path) # read .wav file
        hop_length = math.floor(sr*0.010) # 10ms hop
        win_length = math.floor(sr*0.025) # 25ms frame
        # mfcc is 12 x T matrix
        mfcc = librosa.feature.mfcc(
            y, sr, n_mfcc=12, n_fft=1024,
            hop_length=hop_length, win_length=win_length)
        # substract mean from mfcc --> normalize mfcc
        mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
        # delta feature 1st order and 2nd order
        delta1 = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        # X is 36 x T
        X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
        # return T x 36 (transpose of X)
        return X.T # hmmlearn use T x N matrix
    except:
        pass

def get_class_data(data_dir):
    files = os.listdir(data_dir)
    mfcc = [get_mfcc(os.path.join(data_dir,f)) for f in files if f.endswith(".wav")]
    return mfcc

def clustering(X, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
    kmeans.fit(X)
    print("centers", kmeans.cluster_centers_.shape)
    return kmeans  

def get_dataset():
    # class_names = TRAIN_CLASS_NAMES if mode == 'train' else TEST_CLASS_NAMES
    dataset = {}
    for cname in CLASS_NAMES:
        print(f"Load {cname} dataset")
        dataset[cname] = get_class_data(os.path.join(DATA_PATH, cname))
    return dataset

