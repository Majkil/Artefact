import librosa
import math
import numpy as np
import sklearn
import fastaudio.core.signal as fcs

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

def gmm_features(x,sr ,hop_length):
    audio = librosa.load(x, sr=sr)[0]
    energy = normalize(librosa.feature.rms(audio, hop_length=hop_length)[0])
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
    autocorrelated = librosa.autocorrelate(audio)
    autocorrelated_bandwidth = normalize(librosa.feature.spectral_bandwidth(
        autocorrelated, hop_length=hop_length)[0])
    stack = librosa.util.stack([energy, zcr, autocorrelated_bandwidth], axis=1)
    return stack
