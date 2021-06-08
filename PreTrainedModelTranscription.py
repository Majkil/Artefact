import pydub
import librosa
import os
import numpy as np
import librosa.display as display
import IPython.display as ipd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import fastaudio.core.signal as fcs
from utilities import *
from preprocessing import *
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from pydub import AudioSegment

import os

lvpath ="E:\Datasets\Voice\Librivox\dev\LibriSpeech\dev-clean"

clips =fcs.get_audio_files(lvpath)

clip= clips[380]
transcription = load_clip_transcription(clip)


audio,sr = librosa.load(clip)   



tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

input_values = tokenizer(audio, return_tensors = "pt").input_values

logits = model(input_values).logits

prediction = torch.argmax(logits, dim = -1)
transcription = tokenizer.batch_decode(prediction)[0]
#print("Actual transcription: ", GetTranscription.get_file_transcript(clip))
print("Model Transcription:  ", transcription)

