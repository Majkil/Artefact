# In[2]:
from re import X
from sklearn import preprocessing
from tensorflow.keras import activations
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from utilities import *
from preprocessing import *


lvpath = "E:\Datasets\Voice\Librivox\dev\LibriSpeech\dev-clean"
libri_train = "E:\Datasets\Voice\LibriSpeech"
mcvpath = "E:\Datasets\Voice\Mozilla Common Voice\en\cv-corpus-6.1-2020-12-11\en"
single_word = "./samples/but bowl.wav"

# In[1]:
# import warnings
# warnings.filterwarnings('ignore')
# import keras

# In[3]:
clips = fcs.get_audio_files(libri_train)
print(len(clips))
sr = 16000
hop_length = int(sr/200)
frame_length = int(hop_length*2)
min_duration = hop_length*10
min_voiced_duration_ms = 50
energy_threshold = 0.05

#%%
def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    h = array.shape[0]
    w = array.shape[1]
    a = (xx - h) // 2
    aa = xx - a - h
    b = (yy - w) // 2
    bb = yy - b - w
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


# In[7]:
with open("exact.npy", 'rb') as f:
    exact = np.load(f, allow_pickle=True)
with open("known_clips.npy", 'rb') as f:
    known_clips = np.load(f, allow_pickle=True)


# In[243]:
# labels_all = []
# for x in exact:
#     transcription = load_clip_transcription(x)
#     phonemes = all_phones_to_array(transcription)
#     labels_all.extend(np.unique(phonemes))
# unique_phones = np.unique(labels_all).tolist()
#np.save('unique_phones.npy',unique_phones)
with open("unique_phones.npy", 'rb') as f:
    unique_phones = np.load(f, allow_pickle=True)


# In[9]:
features_count = 20
series_length = 160
features = []
labels = []
skipped = []
#%%
raw_audio= []
#%%
hl_10ms = int(sr/100)
hl_4ms = int(sr/250)
# %% 
# 0 to 12386 no scanning
# 12386  to 14008 with scanning
for x in known_clips[0:500]:
    # x =exact[0]
    transcription = load_clip_transcription(x)
    phonemes = all_phones_to_array(transcription)
    #scan for parameters
    segments , phoneme_sections, sr = process_clip2(x, len(phonemes))
    audio = load_clip(x, sr)
    
    if len(phonemes) != len(phoneme_sections):
        print("\n\nposition: ",known_clips.tolist().index(x))
        print("\nskipped: ", x, len(skipped), "\n")
        skipped.append(x)
        continue

    print("Current segments count: ", len(features))
    for i in range(len(phoneme_sections)):
        raw_audio.append(audio[phoneme_sections[i][0]:phoneme_sections[i][1]])
        mfcc = librosa.feature.mfcc(
            audio[phoneme_sections[i][0]:phoneme_sections[i][1]],
            hop_length=int(hop_length/2), sr=sr, n_fft=hop_length, n_mfcc=features_count)
        try:
            data = np.array([padding(mfcc, features_count, series_length)])
            labels.append(unique_phones.index(phonemes[i]))
            features.append(data)
        except:
            print(phonemes[i], mfcc.shape)


# In[10]:


output = np.array(np.concatenate(features, axis=0))
len(labels), len(
    features), features[0].shape, features[1].shape, features[2].shape, features[2].shape
# for x in range(len(features)):
# print(features[x].shape , labels[x])
features[0].shape[2]
output.shape
# data_con =np.concatenate(np.array(data))[0]


# In[11]:


# Split twice to get the validation set
X_train, X_test, y_train, y_test = train_test_split(
    output, np.array(labels), test_size=0.10, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123)
# Print the shapes
X_train.shape, X_test.shape,len(y_train), len(y_test) ,X_val.shape,  len(y_val)


# In[12]:
input_shape = (features_count, series_length)
model =tfk.Sequential()
#model.add(tfkl.Conv2D(16, kernel_size=(2,2),activation='relu', input_shape=input_shape))
#model.add(tfkl.MaxPooling2D(pool_size=(2,2), padding='same'))
#model.add(tfkl.Reshape((-1,16)))
#model.add(Dense(45, activation = 'relu', input_shape=input_shape))
model.add(LSTM(160,activation='relu', input_shape=input_shape))
#model.add(tfkl.Flatten())
#model.add(Dropout(0.25))
model.add(Dense(180, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(90, activation='relu'))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(45, activation = 'softmax'))
model.compile(loss=tfk.losses.sparse_categorical_crossentropy, metrics=['accuracy'], optimizer=tfk.optimizers.Adam(learning_rate=1.3))
#early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')


# In[16]:
X_train_r = X_train.reshape(X_train.shape[0], features_count, series_length,1)
X_val_r = X_val.reshape(X_val.shape[0], features_count, series_length,1)
X_test_r = X_test.reshape(X_test.shape[0], features_count, series_length,1)
history = model.fit(X_train_r, y_train, epochs=5, batch_size=32,
                    validation_data=(X_val_r, y_val), shuffle=False, verbose=1)

#%%
history = model.fit(X_train, y_train, epochs=3, batch_size=150,
                    validation_data=(X_val, y_val), shuffle=False, verbose=1)
#%%
test_loss , test_acc = model.evaluate(X_test, y_test, verbose=2)
test_loss, test_acc

# In[14]:
model.summary()
#%%
model.save('model_checkpoint')

# In[ ]:
model_attempt_1 = model


# %%

pred = model.predict(X_test[1:2])

# %%
history_dict=history.history
loss_values=history_dict['loss']
acc_values=history_dict['acc']
val_loss_values = history_dict['val_loss']
val_acc_values=history_dict['val_acc']
epochs=range(1,16)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
ax1.plot(epochs,loss_values,'co',label='Training Loss')
ax1.plot(epochs,val_loss_values,'m', label='Validation Loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(epochs,acc_values,'co', label='Training accuracy')
ax2.plot(epochs,val_acc_values,'m',label='Validation accuracy')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.show()
# %%
#np.save('checkpoint_features_position_2306_to_6727.npy', output)
np.save('checkpoint_labels_position_2306_to_6727.npy',labels)
#%%
with open("checkpoint_features_position_2306_to_6727.npy", 'rb') as f:
    output_2 = np.load(f, allow_pickle=True)
with open("checkpoint_labels_position_2306_to_6727.npy", 'rb') as f:
    labels_temp = np.load(f, allow_pickle=True)
    labels_temp = labels.tolist()
with open("raw_audio_Segments_2306.npy", 'rb') as f:
    raw_audio = np.load(f, allow_pickle=True)
    raw_audio = raw_audio.tolist()
#%%
np.save('raw_audio_Segments_2306.npy', raw_audio)
# %%
with open("features_position_12376.npy", 'rb') as f:
    features_old = np.load(f, allow_pickle=True)
with open("labels_position_12376.npy", 'rb') as f:
    labels_old = np.load(f, allow_pickle=True)
with open("features_position_12376_to_14008.npy", 'rb') as f:
    features_new = np.load(f, allow_pickle=True)
with open("labels_position_12376_to_14008.npy", 'rb') as f:
    labels_new = np.load(f, allow_pickle=True)



# %%
output = np.concatenate((features_old, features_new),axis=0)
labels = np.concatenate((labels_old,labels_new),axis =0)

#%%

with open("checkpoint_features_position_2306.npy", 'rb') as f:
    output = np.load(f, allow_pickle=True)
# %%
with open("checkpoint_labels_position_2075.npy", 'rb') as f:
    labels = np.load(f, allow_pickle=True)
# %%
