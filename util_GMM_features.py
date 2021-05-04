from utilities import *


def gmm_features(x, sr, hop_length):
    energy = normalize(librosa.feature.rms(x, hop_length=hop_length)[0])
    zcr = librosa.feature.zero_crossing_rate(x, hop_length=hop_length)[0]
    autocorrelated = librosa.autocorrelate(x)
    autocorrelated_bandwidth = normalize(librosa.feature.spectral_bandwidth(
        autocorrelated, hop_length=hop_length)[0])
    stack = librosa.util.stack([energy, zcr, autocorrelated_bandwidth], axis=1)
    return stack
