import sklearn
import librosa
import librosa.display as display
import numpy as np
import fastaudio.core.signal as fcs
import fastaudio.augment.preprocess as fap
import fastaudio.augment.spectrogram as fas
import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy import signal
from scipy.io import wavfile
import IPython.display as ipd
from joblib import dump, load
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import textdistance

textdistance.jaro_winkler("they are going to school", "school")

tokenizer = Wav2Vec2Tokenizer.from_pretrained(
        "facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


def transcribe_audio_fb(audio):
    # tokenizer = Wav2Vec2Tokenizer.from_pretrained(
    #     "facebook/wav2vec2-base-960h")
    # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    input_values = tokenizer(audio, return_tensors="pt").input_values

    logits = model(input_values).logits

    prediction = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(prediction)[0]
    #print("Actual transcription: ", GetTranscription.get_file_transcript(clip))
    #print("Model Transcription:  ", transcription)
    return transcription
