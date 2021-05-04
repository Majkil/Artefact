#!/usr/bin/env python
# coding: utf-8

# In[4]:
# region  imports and consts
from joblib import dump, load
import librosa
import math
import librosa.display as display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.mixture import GaussianMixture
import fastaudio.core.signal as fcs
from util_GMM_features import gmm_features
#import GetTranscription
lvpath = "E:\Datasets\Voice\Librivox\dev\LibriSpeech\dev-clean"
libri_train = "E:\Datasets\Voice\LibriSpeech"
mcvpath = "E:\Datasets\Voice\Mozilla Common Voice\en\cv-corpus-6.1-2020-12-11\en"
single_word = "./samples/but bowl.wav"

# endregion

# In[3]:
sr = 8000
hop_length = int(sr/1000)
covariance_type="spherical"
samples = 2000
clips = fcs.get_audio_files(libri_train)

print("sr: ",sr,"   hop_length: ",hop_length,"  covariance_type:",covariance_type)
# In[3]:
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

EM = GaussianMixture(n_components=2, covariance_type=covariance_type)
counter = 0
for x in clips[0:samples]:
    print(counter, x)
    clip, sr = librosa.load(x, sr=sr)
    s = gmm_features(clip,sr,hop_length)
    EM.fit(s)
    counter += 1
filename = f"EM2c_samples-{samples}_covar-{covariance_type}_hopLength-{hop_length}_sr-{sr}.joblib"

#dump(EM, 'EM_samples2k_covar-spherical_hopLength-60_sr-12k.joblib')
dump(EM,filename)
print("sr: ",sr,"   hop_length: ",hop_length,"  covariance_type:",covariance_type)