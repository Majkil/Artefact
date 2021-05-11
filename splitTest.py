# In[]:
from energy_split import *
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

lvpath = "E:\Datasets\Voice\Librivox\dev\LibriSpeech\dev-clean"
libri_train = "E:\Datasets\Voice\LibriSpeech"
mcvpath = "E:\Datasets\Voice\Mozilla Common Voice\en\cv-corpus-6.1-2020-12-11\en"
single_word = "./samples/but bowl.wav"

clips = fcs.get_audio_files(libri_train)
sr = 16000
hop_length = int(sr/200)
frame_length = int(hop_length*2)
clips = fcs.get_audio_files(libri_train)
clip = clips[2402]
audio, sr = librosa.load(clip, sr=sr)
print(clip)
print(GetTranscription.get_file_transcript(clip))
print(audio.shape)
#In[]
threshold = 0.5
segments = Split(audio,hop_length,frame_length,min_duration=3, energy_threshold=0.05)
segment = segments[1]
print(segments)
print(len(segments))
print(segment)

# In[]

part = audio[segment[0]*hop_length:segment[1] * hop_length]
segments = Split2(part, hop_length, frame_length, sr, min_duration=hop_length*8)
print(segments)
print(len(segments))

# %%
