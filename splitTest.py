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
from preprocessing import *
lvpath = "E:\Datasets\Voice\Librivox\dev\LibriSpeech\dev-clean"
libri_train = "E:\Datasets\Voice\LibriSpeech"
mcvpath = "E:\Datasets\Voice\Mozilla Common Voice\en\cv-corpus-6.1-2020-12-11\en"
single_word = "./samples/but bowl.wav"

clips =fcs.get_audio_files(libri_train)
len(clips)
sr = 16000
hop_length = int(sr/200)
frame_length = int(hop_length*2)
min_duration=hop_length*10
min_voiced_duration_ms = 50 
energy_threshold = 0.05

clip = clips[22]
audio, sr = librosa.load(clip, sr=sr)
print(clip)
print(GetTranscription.get_file_transcript(clip))
print(audio.shape)
#In[]
threshold = 0.5
segments = Split(audio,hop_length,frame_length,min_duration=20,sr=sr, energy_threshold=0.04)
segment = segments[0]
print(segments)
print(len(segments))
print(segment)

# In[]
part = audio[segment[0]:segment[1]]
segments = Split2(part, hop_length, frame_length, sr, min_duration=hop_length*10)
print(segments)
print(len(segments))

#In[]:
all_bits = []
for segment in segments:
    starting = segment[0]
    segment_boundaries = Split2(audio[starting:segment[1]], hop_length=hop_length, frame_length=frame_length, sr= sr, min_duration=500)
    for bit in segment_boundaries:
        #print((segment[0]+bit[0]), (segment[0]+bit[1]) )
        x1 =starting+bit[0]
        x2 =starting+bit[1]
        b = (x1,x2)
        #print(type(b))
        #print(segment[0]+bit[0],segment[0]+bit[1] )
        all_bits.append(b)
    #print(segment, "   ",segment_boundaries)
print(all_bits)
# %%
transcription = GetTranscription.get_file_transcript(clip)
from m_dictionary import *
sectionphones = get_phonemes_for_sentence(sentence = transcription)
phone_array = []
for word in sectionphones:
    for char in word[0]:
        if char != "ˈ" and char != '\u200d' and char != "/" and char != " " and char != ",":
            if char =='ː':
                phone_array[-1] = phone_array[-1]+char
            else:
                phone_array.append(char)
print(phone_array)
print(transcription)
print(sectionphones)
len(phone_array)
#%%
test = "ɐbɒlɪʃɪ"
MatchPhonesToText(test)

#%%
for word in sectionphones:
    if len(word) == word.count(' '):
        print(len(word))
        print(word)
        continue
# %%
clip_address = clips[22]
transcription = load_clip_transcription(clip_address)
phonemes_in_clip_transcription = all_phones_to_array(transcription)
#%%
if 'XXXXXX' not in phonemes_in_clip_transcription:
    #print(counter , clip_address)
    # not all words are in the dictionary 
    audio = load_clip(clip_address, sr)
    segments = split_into_segments(audio,hop_length, frame_length, sr, min_voiced_duration_ms, energy_threshold)
    ##clip_from_segments(segments)
    phoneme_sections = all_phoneme_Sections_in_clip(audio, segments, sr, frame_length, hop_length, min_duration)
    print(len(phoneme_sections))
    print(len(phonemes_in_clip_transcription))


# %%
pp = process_clip(clip_address, len(phonemes_in_clip_transcription))
