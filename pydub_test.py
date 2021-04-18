import pydub
from pydub.silence import split_on_silence
import os
import numpy as np

print(os.getcwd()+"\\ffmpeg\\bin")

pydub.AudioSegment.converter =str(os.getcwd()+"\\ffmpeg\\bin")
from pydub import AudioSegment
asd = AudioSegment.from_wav(str(os.getcwd()+"\\" +"84-121123-0000.wav"))
chunks = split_on_silence(asd,min_silence_len=10,silence_thresh=22)

print(chunks)


def split_into_words(arr):
    minimum_speach_lenght = 28
    treshold = np.std(arr, dtype=np.float64)*1.1
    print(treshold)

    speachy_indices = []
    for i, v in enumerate(arr):
        if v > treshold:
            speachy_indices.append(i)
   
    speachy_sections = []
    a_temp_list = [speachy_indices[0]]  

    for i, v in enumerate(speachy_indices[1:]):
        if v == a_temp_list[-1]+1:
            a_temp_list.append(v)
        else:
            speachy_sections.append(a_temp_list)
            a_temp_list = [v]
   
#     drop intervals that are too short
    clean = [a for a in speachy_sections if len(a)>minimum_speach_lenght]
    sound_sections = [[min(a), max(a)] for a in clean]
   
    print(f"soundSections = {sound_sections}")
   
    snipping_indices = []
    start, end = 0, 0
    for previous, current in zip(sound_sections, sound_sections[1:]):      
        end = int((previous[1]+current[0])/2)
        snipping_indices.append([start,end])
        start = end+1
   
    snipping_indices.append([start,len(arr)])
   
    arrays =  [arr[split[0]:split[1]] for split in snipping_indices]
    return arrays


