#!/usr/bin/env python
# coding: utf-8

# In[4]:
# region  imports and consts
from joblib import dump, load
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
from util_GMM_features import gmm_features
lvpath = "E:\Datasets\Voice\Librivox\dev\LibriSpeech\dev-clean"
libri_train = "E:\Datasets\Voice\LibriSpeech"
mcvpath = "E:\Datasets\Voice\Mozilla Common Voice\en\cv-corpus-6.1-2020-12-11\en"
single_word = "./samples/but bowl.wav"
model1 ="EM_samples2k_covar-spherical_hopLength-60_sr-12k.joblib"
model2 ="EM_samples-1000_covar-spherical_hopLength-50_sr-10000.joblib"
model3 ="EM_samples-2000_covar-spherical_hopLength-20_sr-8000.joblib"
model4 ="EM_samples-2000_covar-spherical_hopLength-80_sr-8000.joblib"
model5 ="EM_samples-2000_covar-spherical_hopLength-80_sr-16000.joblib"
model6 ="EM_samples-4000_covar-spherical_hopLength-40_sr-8000.joblib"
model7 ="EM_samples-2000_covar-spherical_hopLength-16_sr-16000.joblib"
model8="EM2c_samples-2000_covar-spherical_hopLength-8_sr-8000.joblib"
# endregion

# In[3]:

clips = fcs.get_audio_files(libri_train)
clip = clips[6701]
EM = load(model7)
sr = 16000
hop_length = int(sr/1000)
audio = librosa.load(clip, sr=sr)[0]

# In[3]:
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

#region Test

# In[1]:
three = gmm_features(audio,sr,hop_length)
print(three)
print(three.shape)

x = EM.predict(three)

# In[215]:
plt.figure(figsize=(21, 9))
#raudio= librosa.resample(y=audio, orig_sr=sr, target_sr=100)

for s in range(len(x)):
    if x[s]==2:
        plt.axvline(x=s*hop_length, ymin=-0.4, ymax=0.6, c='r')
    if x[s]==1:
        plt.axvline(x=s*hop_length, ymin=-0.4, ymax=0.6, c='y')
    if x[s]==0:
        plt.axvline(x=s*hop_length, ymin=-0.4, ymax=0.6, c='g')
plt.plot(audio)

#endregion


#In[]:
groups = []
start=0
for z in range(len(x)):
    if not x[z]==x[z-1]:
        groups.append([start,int(z*hop_length), x[z-1]])
        start= (z*hop_length)+1
        


# In[177]:


fig = plt.figure(figsize=(16, 16))
plt.scatter(three[x == 0, 0], three[x == 0, 1], c='r')
plt.scatter(three[x == 1, 0], three[x == 1, 1], c='b')
plt.scatter(three[x == 2, 0], three[x == 2, 1], c='y')
plt.xlabel('Zero Crossing Rate (scaled)')
plt.ylabel('Energy (scaled)')
plt.legend(('Class 0', 'Class 1'))


# In[174]:

fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(projection='3d')
plt.scatter(three[x == 0, 0], three[x == 0, 1], three[x == 0, 2], c='r')
plt.scatter(three[x == 1, 0], three[x == 1, 1], three[x == 1, 2], c='b')
plt.scatter(three[x == 2, 0], three[x == 2, 1], three[x == 2, 2], c='y')
plt.xlabel('Zero Crossing Rate (scaled)')
plt.ylabel('Energy (scaled)')
plt.legend(('Class 0', 'Class 1'))
# In[169]:
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(projection='3d')
plt.scatter(three[0], three[1], three[2])


# In[170]:


model = sklearn.cluster.KMeans(n_clusters=3)
labels = model.fit_predict(three)

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(projection='3d')
plt.scatter(three[labels == 0, 0], three[labels == 0, 1],
            three[labels == 0, 2], c='b')
plt.scatter(three[labels == 1, 0], three[labels == 1, 1],
            three[labels == 1, 2], c='r')
plt.scatter(three[labels == 2, 0], three[labels == 2, 1],
            three[labels == 2, 2], c='g')
plt.xlabel('Zero Crossing Rate (scaled)')
plt.ylabel('Energy (scaled)')
plt.legend(('Class 0', 'Class 1'))






# In[147]:


normalize(three[:, 1])
np.argmax(three[:, 0])
np.set_printoptions(formatter={'int': lambda x: "{0:0.3f}".format(x)})
print(three[:, 0])


# In[107]:


plt.plot(three[x == 2, :])


# In[178]:


print(len(x))
print(x)
x[0:150]






# In[ ]:


plt.figure(figsize=(16, 9))
scaled_audio = (sklearn.preprocessing.minmax_scale(audio, axis=0))
audio_range = np.max(scaled_audio) - np.min(scaled_audio)
mean = np.mean(scaled_audio)
print(audio_range)
y = np.full(len(audio), mean-audio_range*0.01)  # audio non silence min
y1 = np.full(len(audio), mean+audio_range*0.01)  # audio non silence max
y2 = np.full(len(audio), 0.8)
y3 = np.full(len(audio), 0.15)
energy = librosa.pcen(audio)
delta_energy = librosa.feature.delta(energy)
delta_energy2 = librosa.feature.delta(delta_energy)

plt.plot(sklearn.preprocessing.minmax_scale(audio, axis=0))  # blue
plt.plot(sklearn.preprocessing.minmax_scale(energy, axis=0))  # yellow
plt.plot(sklearn.preprocessing.minmax_scale(delta_energy, axis=0))  # green
plt.plot(y, c='r')
plt.plot(y1, c='r')
plt.plot(y2, c='g')
plt.plot(y3, c='g')
#plt.plot(sklearn.preprocessing.minmax_scale(delta_energy2, axis=0))


# In[183]:


def split_by_energy(audio):
    frames = len(audio)
    # energy per frame
    energy = librosa.pcen(audio)
    # rate of change of energy
    delta_energy = librosa.feature.delta(energy)
    # rate of change of change of energy
    delta_energy2 = librosa.feature.delta(delta_energy)

    s_audio = sklearn.preprocessing.minmax_scale(audio, axis=0)
    s_energy = sklearn.preprocessing.minmax_scale(energy, axis=0)
    s_d_energy = sklearn.preprocessing.minmax_scale(delta_energy, axis=0)
    s_d_2_energy = sklearn.preprocessing.minmax_scale(delta_energy2, axis=0)

    audio_range = np.max(s_audio) - np.min(s_audio)
    print(audio_range)
    mean = np.mean(s_audio)

    #print("scaled delta energy less than 0.5 ", np.count_nonzero( s_d_energy<0.8))
    #print("scaled audio less than 0.5 ", np.count_nonzero(0.45 >s_audio or s_audio> 0.55))

    out = []
    # blue audio
    # yellow energy
    # green de1
    # red de2
    for x in range(frames):
        if s_audio[x] > (mean+0.01) or s_audio[x] < (mean-0.01):
            if s_d_energy[x] < 0.8:
                out.append(x)
    return out


# In[184]:
splits = split_by_energy(audio[19800:27060])
print(len(splits))
print(splits)

# In[187]:
##
plt.figure(figsize=(16, 9))
energy = librosa.pcen(audio[19800:27060])
# plt.plot(audio) #-0.4-0.6
plt.plot(normalize(energy))
plt.plot(normalize(audio[19800:27060]))

for x in split_by_energy(audio[19800:27060]):
    plt.axvline(x=x, ymin=-1, ymax=1, label=str(x), c='g')
for x in energy:
    if math.sqrt(x**2) < 0.02:
        plt.axvline(x=x, ymin=-1, ymax=1, label=str(x), c='r')
plt.show()


# In[253]:



# In[ ]:


# In[15]:


energy = normalize(librosa.pcen(audio))
delta_energy = normalize(librosa.feature.delta(energy))
delta_energy2 = normalize(librosa.feature.delta(delta_energy))

plt.figure(figsize=(16, 9))
plt.plot(audio[:])  # -0.4-0.6
plt.plot(energy)  # -0.4-0.6
plt.plot(delta_energy[:])  # 0- -30
plt.plot(delta_energy2[:])  # 0-2
plt.show()

print(energy)
print(delta_energy)
#print (min(energy))


# In[ ]:



# In[246]:


stft = librosa.stft(e[1781:4638], hop_length=220)
print(stft.shape)
spectogram = np.abs(stft)

log_spectogram = librosa.amplitude_to_db(spectogram)
spectogram

plt.figure(figsize=(21, 9))
librosa.display.specshow(log_spectogram, y_axis='log')
plt.xlabel("Time")
plt.ylabel("Freq")
plt.colorbar()
plt.show()




# In[153]:


x = audio
spectral_centroids = librosa.feature.spectral_centroid(
    audio, sr=sr, hop_length=220)[0]
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(
    x+0.01, sr=sr, hop_length=220)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(
    x+0.01, sr=sr, p=3,  hop_length=220)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(
    x+0.01, sr=sr, p=4,  hop_length=220)[0]
#spectral_bandwidth_5 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=5, hop_length=220)[0]
#spectral_bandwidth_6 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=6, hop_length=220)[0]
plt.figure(figsize=(15, 12))
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
plt.plot(t, normalize(spectral_bandwidth_3), color='g')
plt.plot(t, normalize(spectral_bandwidth_4), color='y')
#plt.plot(t, normalize(spectral_bandwidth_5), color='b')
#plt.plot(t, normalize(spectral_bandwidth_6), color='pink')
plt.legend(('p = 2', 'p = 3', 'p = 4'))




dump(EM, 'EM200tied.joblib')


# In[148]:


a_file = open("test.txt", "w")
np.savetxt(a_file, three)
a_file.close()


# In[ ]:
