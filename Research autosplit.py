#!/usr/bin/env python
# coding: utf-8

# In[2]:


from utilities import *
from preprocessing import *
lvpath = "E:\Datasets\Voice\Librivox\dev\LibriSpeech\dev-clean"
libri_train = "E:\Datasets\Voice\LibriSpeech"
mcvpath = "E:\Datasets\Voice\Mozilla Common Voice\en\cv-corpus-6.1-2020-12-11\en"
single_word = "./samples/but bowl.wav"


clips = fcs.get_audio_files(libri_train)
print(len(clips))
sr = 22000
hop_length = int(sr/200)
frame_length = int(hop_length*2.5)
min_duration = hop_length*10
min_voiced_duration_ms = 50
energy_threshold = 0.05


# In[3]:


c=182
clip = clips[c]
transcript = load_clip_transcription(clip)
phonemes =all_phones_to_array(transcript)
print(transcript ,"\n || word count: ", len(transcript.split(" ")), "|| phoneme count:", len(phonemes))
audio,sr = librosa.load(clip, sr)
hl = int(sr/200)
fl= int(hl*2)
segments = split_segments(audio, hl, fl ,sr)
print("segmented words:",len(segments))
all_bits = []
for seg in segments:
    s,e= seg[0], seg[1]
    seg_audio = audio[s:e]
    boundaries = phonme_boundaries(seg_audio,sr)
    if len(boundaries)>1:
        bits = boundaries_to_segments(boundaries)
        for b in range(len(bits)-1):
            all_bits.append(audio[s+bits[b][0]:s+bits[b][1]])

print("detected phonemes:", len(all_bits))


# In[4]:


seg= segments[0]
s,e= seg[0], seg[1]
segment =audio[s:e]
hop_length = int(sr / 200)
min_duration = hop_length*10
frame_length = int(hop_length * 2.5)
mins = []
premph_seg = librosa.effects.preemphasis(segment)
sec_energy = librosa.feature.rms(
    np.abs(premph_seg), hop_length=hop_length, frame_length=frame_length)[0]

mins.extend(signal.argrelmin(sec_energy)[0] * hop_length)
#if len(mins) <2:
    #return [()]
temp_mins = [mins[0]]
means = []
for x in range(0, len(mins) - 1):
    means.append(np.mean(segment[mins[x]:mins[x + 1]]))
    # print(np.mean(segment[mins[x]:mins[x+1]]))
means = normalize(means)
for x in range(1, len(means)):
    diff = np.abs(means[x] - means[x-1])
    if diff >= 0.1 :
        temp_mins.append(mins[x])
mins = temp_mins
temp_mins =[mins[0]]
for m in range(len(mins)-1):
    if mins[m+1]-temp_mins[-1]>= min_duration:
        temp_mins.append(mins[m+1])
mins= temp_mins
#mins = np.array(mins)


# In[5]:


#print(len(segment),mins ,mins[1]-mins[0])
audio , sr = librosa.load(clips[213],sr)
ipd.Audio(audio,rate=sr)


# In[24]:


from tqdm import tqdm
#words_actual, words_predicted, phonemes_actual, phonemes_predicted , phonemes , Pc ,Po= [],[],[],[],[],[],[]
for c in tqdm(range(8983, len(clips))):
     words_actual_t, words_predicted_t, phonemes_actual_t, phonemes_predicted_t , phonemes_t= auto_process(clips[c])
     words_actual.append(words_actual_t)
     words_predicted.append(words_predicted_t)
     phonemes_actual.append(phonemes_actual_t)
     phonemes_predicted.append(phonemes_predicted_t)
     #phonemes.append(phonemes_t)
     Pc.append(words_predicted_t/words_actual_t)
     if phonemes_predicted_t<0:
        Po.append((phonemes_predicted_t-phonemes_actual_t)/phonemes_predicted_t)
     else:
         Po.append(0)


# In[27]:


np.save("./segmentation/words_actual.npy", words_actual)
np.save("./segmentation/words_predicted.npy", words_predicted)
np.save("./segmentation/phonemes_actual.npy", phonemes_actual)
np.save("./segmentation/phonemes_predicted.npy", phonemes_predicted)
np.save("./segmentation/phonemes.npy", phonemes)
np.save("./segmentation/Pc.npy", Pc)
np.save("./segmentation/Po.npy", Po)
# np.array([words_actual, words_predicted, phonemes_actual, phonemes_predicted , phonemes , Pc ,Po])


# In[22]:


#np.average(Pc), np.average(Po)
Po
for c in range(len(Po)):
    if phonemes_predicted[c] == 0:
        print(c)
    else:
        Po[c]= (phonemes_predicted[c]-phonemes_actual[c])/phonemes_predicted[c]
#(phonemes_predicted_t-phonemes_actual_t)/phonemes_predicted_t


# In[23]:


np.average(Po)

