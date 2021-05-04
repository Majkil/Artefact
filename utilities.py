import sklearn
import librosa
import librosa.display as display
import numpy as np
import fastaudio.core.signal as fcs
import fastaudio.augment.preprocess as fap
import fastaudio.augment.spectrogram as fas
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
import IPython.display as ipd
from joblib import dump, load


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)