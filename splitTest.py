#In[]:
import librosa
import math
import librosa.display as display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.mixture import GaussianMixture
import fastaudio.core.signal as fcs
import GetTranscription

#In[]:
lvpath ="E:\Datasets\Voice\Librivox\dev\LibriSpeech\dev-clean"
libri_train ="E:\Datasets\Voice\LibriSpeech"
mcvpath ="E:\Datasets\Voice\Mozilla Common Voice\en\cv-corpus-6.1-2020-12-11\en"
single_word = "./samples/but bowl.wav"

clips =fcs.get_audio_files(libri_train)
print(len(clips))
clip= clips[6576]
audio,sr = librosa.load(clip)
print(clip)
print(GetTranscription.get_file_transcript(clip))
print(audio.shape)

#In[]:

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
# %%
