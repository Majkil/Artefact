# %% Imports
from tqdm import tqdm
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
sr = 22000
hop_length = int(sr/200)
frame_length = int(hop_length*2.5)
min_duration = hop_length*10
min_voiced_duration_ms = 50
energy_threshold = 0.05

# %%


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
#     cription = load_clip_transcription(x)trans
#     phonemes = all_phones_to_array(transcription)
#     labels_all.extend(np.unique(phonemes))
# unique_phones = np.unique(labels_all).tolist()
# np.save('unique_phones.npy',unique_phones)
with open("unique_phones.npy", 'rb') as f:
    unique_phones = np.load(f, allow_pickle=True)
    unique_phones = unique_phones.tolist()

# %%
raw_audio = []
# %%
hl_10ms = int(sr/100)
hl_4ms = int(sr/250)
phoneme_audio = []
phoneme_audio_labels = []
# %%
# processing data using the fb pretrain asr model
for f in clips[7247:9000]:
    
    bits, bit_labels = process_clip_with_fb(f)
    phoneme_audio.extend(bits)
    phoneme_audio_labels.extend(bit_labels)
    print("\nextracted : ", len(bits), " current total:", len(phoneme_audio))
    print("position: ", clips.index(f), "\n")
# %%
np.save("fb_raw_audio2.npy", phoneme_audio)
np.save("fb_raw_audio_labels2.npy", phoneme_audio_labels)

# %%

with open("fb_raw_audio.npy", 'rb') as f:
    phoneme_audio = np.load(f, allow_pickle=True).tolist()
with open("fb_raw_audio_labels.npy", 'rb') as f:
    phoneme_audio_labels = np.load(f, allow_pickle=True).tolist()

#%%

#Prepare raw audio to mfcc
#this will be the sound clip corresponding to the label
features_count = 24
series_length = 165
features = []
labels = []
skipped = []
phone_to_mfcc =[]

for i in tqdm(range(len(phoneme_audio))):
    temp_audio = librosa.effects.preemphasis(phoneme_audio[i])
    mfcc = librosa.feature.mfcc(
        temp_audio, hop_length=hop_length, sr=sr, n_mfcc=features_count)
    try:
        data = np.array([padding(mfcc, features_count, series_length)])
        labels.append(unique_phones.index(phoneme_audio_labels[i]))
        features.append(data)
        phone_to_mfcc.append(phoneme_audio[i])

    except:
        print(phoneme_audio_labels[i], mfcc.shape, len(phoneme_audio[i])/sr)

# In[10]:

output = np.array(np.concatenate(features, axis=0))
len(labels), len(
    features), features[0].shape, features[1].shape, features[2].shape, features[2].shape
# for x in range(len(features)):
# print(features[x].shape , labels[x])
features[0].shape[2]
output.shape



# In[11]:
#First split of data (not filtered)
# Split twice to get the validation set
X_train, X_test, y_train, y_test = train_test_split(
    output, np.array(labels), test_size=0.10, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=123)
# Print the shapes
X_train.shape, X_test.shape, len(y_train), len(
    y_test), X_val.shape,  len(y_val)

# %%
input_shape = (features_count, series_length)
model = tfk.Sequential()
model.add(LSTM(128, input_shape=input_shape))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(55, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(len(unique_phones), activation='softmax'))
# %%
model.compile(loss=tfk.losses.sparse_categorical_crossentropy, metrics=[
              'accuracy'], optimizer=tfk.optimizers.Adam(learning_rate=1.7))
model.summary()

# %%
history = model.fit(X_train, y_train, epochs=10, batch_size=150,
                    validation_data=(X_val, y_val), shuffle=True, verbose=1)
# %%
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
test_loss, test_acc

# %%
accurate_f = []
accurate_labels = []
accurate_raw =[]
# %%
# use model to filter data
for i in tqdm(range(len(output))):
    p1 = model.predict([output[i:i+1]])
    #first order prediction
    
    if np.argmax(p1) == labels[i]:
        accurate_f.append(output[i:i+1])
        accurate_labels.append(labels[i])
        accurate_raw.append(phone_to_mfcc[i])
    #second order prediction
    p1[0][np.argmax(p1)] =0
    if np.argmax(np.delete(p1, np.argmax(p1))) == labels[i]:
        accurate_f.append(output[i:i+1])
        accurate_labels.append(labels[i])
        accurate_raw.append(phone_to_mfcc[i])
    # #third order prediction
    # p1[0][np.argmax(p1)] =0
    # if np.argmax(np.delete(p1, np.argmax(p1))) == labels[i]:
    #     accurate_f.append(output[i:i+1])
    #     accurate_labels.append(labels[i])
    #     accurate_raw.append(phone_to_mfcc[i])


#%%
#Split filtered data
output_2 = np.array(np.concatenate(accurate_f, axis=0))
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    output_2, np.array(accurate_labels), test_size=0.10, random_state=123)
X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(
    X_train_2, y_train_2, test_size=0.25, random_state=123)
# Print the shapes
X_train_2.shape, X_test_2.shape, len(y_train_2), len(
    y_test_2), X_val_2.shape,  len(y_val_2)


input_shape = (features_count, series_length)
filtered_model = tfk.Sequential()
filtered_model.add(LSTM(160, input_shape=input_shape))
filtered_model.add(Dropout(0.2))
filtered_model.add(Dense(160, activation='relu'))
filtered_model.add(Dropout(0.2))
filtered_model.add(Dense(80, activation='relu'))
filtered_model.add(Dropout(0.4))
filtered_model.add(Dense(70, activation='relu'))
filtered_model.add(Dropout(0.4))
filtered_model.add(Dense(60, activation='relu'))
filtered_model.add(Dropout(0.4))
filtered_model.add(Dense(55, activation='relu'))
filtered_model.add(Dropout(0.4))
filtered_model.add(Dense(len(unique_phones), activation='softmax'))

filtered_model.compile(loss=tfk.losses.sparse_categorical_crossentropy, metrics=[
              'accuracy'], optimizer=tfk.optimizers.Adam(learning_rate=1.7))
filtered_model.summary()

# %%
# train model with filtered data
history = filtered_model.fit(X_train_2, y_train_2, epochs=10, batch_size=150,
                    validation_data=(X_val_2, y_val_2), shuffle=True, verbose=1)
test_loss, test_acc = filtered_model.evaluate(X_test_2, y_test_2, verbose=2)
test_loss, test_acc

#%%
model.save('model_fb_lstm_37_3rd_order')





#%%
with open("checked.npy", 'rb') as f:
    checked = np.load(f, allow_pickle=True)
    
with open("checked_labels.npy", 'rb') as f:
    check_labels = np.load(f, allow_pickle=True)
    
#%%
car = librosa.load('./car.wav',22000)
car_split = Split3(car[0], hop_length,sr,min_duration=750)
toTest = []
for x in range(3):
    temp_audio = librosa.effects.preemphasis(car[0][car_split[x][0]:car_split[x][1]])
    mfcc_checked1 = librosa.feature.mfcc(
            temp_audio, hop_length=hop_length, sr=sr, n_mfcc=features_count)
    mfcc_checked1
    data_checked1 = np.array([padding(mfcc_checked1, features_count, series_length)])
    toTest.append(data_checked1)
toTest = np.concatenate(toTest)
#%%
pc1 = filtered_model.predict([toTest[0:3]])

print(unique_phones[np.argmax(pc1)] , check_labels[70])

pc1[0][np.argmax(pc1[0])] = 0
print(unique_phones[np.argmax(pc1)] , check_labels[70])
pc1[0][np.argmax(pc1)] =0
print(unique_phones[np.argmax(pc1)] , check_labels[70])
print( np.argmax(pc1))



#%%

for u in unique_phones:
    print("phoneme",u ,"count:", accurate_labels.count( unique_phones.index(u )) , "label: ", unique_phones.index(u))






# In[12]:
input_shape = (features_count, series_length, 1)
model = tfk.Sequential()
model.add(tfkl.Conv1D(16, kernel_size=3,
                      activation='relu', input_shape=input_shape))
model.add(tfkl.MaxPooling2D(pool_size=3, padding='same'))
model.add(tfkl.Conv1D(16, kernel_size=3, activation='relu'))
model.add(tfkl.MaxPooling2D(pool_size=3, padding='same'))
model.add(tfkl.Reshape((-1, 16)))
model.add(LSTM(128, input_shape=input_shape))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(len(unique_phones), activation='softmax'))
# %%

model.compile(loss=tfk.losses.sparse_categorical_crossentropy, metrics=[
              'accuracy'], optimizer=tfk.optimizers.Adam(learning_rate=2.3))
#early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')


# In[16]:
X_train_r = X_train.reshape(X_train.shape[0], features_count, series_length, 1)
X_val_r = X_val.reshape(X_val.shape[0], features_count, series_length, 1)
X_test_r = X_test.reshape(X_test.shape[0], features_count, series_length, 1)
history = model.fit(X_train_r, y_train, epochs=7, batch_size=32,
                    validation_data=(X_val_r, y_val), shuffle=False, verbose=1)

# X_test_r[1:2].shape



#%%
np.save("ai_Selected_mfcc.npy",output )
np.save("ai_Selected_labels.npy",accurate_labels )



# %%
history = model.fit(X_train_2, y_train_2, epochs=10, batch_size=150,
                    validation_data=(X_val_2, y_val_2), shuffle=True, verbose=1)
# %%
test_loss, test_acc = model.evaluate(X_test_2, y_test_2, verbose=2)
test_loss, test_acc




# In[14]:
model.summary()
# %%
model.save('model_fb_lstm_52')

# In[ ]:
model_attempt_1 = model


# %%

pred = model.predict(X_test[1:2])

# %%
history_dict = history.history
loss_values = history_dict['loss']
acc_values = history_dict['acc']
val_loss_values = history_dict['val_loss']
val_acc_values = history_dict['val_acc']
epochs = range(1, 16)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(epochs, loss_values, 'co', label='Training Loss')
ax1.plot(epochs, val_loss_values, 'm', label='Validation Loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(epochs, acc_values, 'co', label='Training accuracy')
ax2.plot(epochs, val_acc_values, 'm', label='Validation accuracy')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.show()
# %%
np.save('raw_audio_new_1700_5000.npy', raw_audio)
np.save('labels_new_1700_5000.npy', labels)
np.save('features_new_1700_5000.npy', features)
# %%
with open("checkpoint_features_position_2306_to_6727.npy", 'rb') as f:
    output_2 = np.load(f, allow_pickle=True)
with open("checkpoint_labels_position_2306_to_6727.npy", 'rb') as f:
    labels_temp = np.load(f, allow_pickle=True)
    labels_temp = labels.tolist()
with open("raw_audio_Segments_2306.npy", 'rb') as f:
    raw_audio = np.load(f, allow_pickle=True)
    raw_audio = raw_audio.tolist()
# %%
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
output = np.concatenate((features_old, features_new), axis=0)
labels = np.concatenate((labels_old, labels_new), axis=0)

# %%

with open("checkpoint_features_position_2306.npy", 'rb') as f:
    output = np.load(f, allow_pickle=True)
# %%
with open("checkpoint_labels_position_2075.npy", 'rb') as f:
    labels = np.load(f, allow_pickle=True)


# %%
#deprecated 
for x in known_clips[5000:-1]:
    # x =exact[0]
    transcription = load_clip_transcription(x)
    phonemes = all_phones_to_array(transcription)
    # scan for parameters
    segments, phoneme_sections, sr = process_clip2(x, len(phonemes))
    audio = load_clip(x, sr)

    if len(phonemes) != len(phoneme_sections):
        print("\n\nposition: ", known_clips.tolist().index(x))
        #print("\nskipped: ", x, len(skipped), "\n")
        skipped.append(x)
        continue

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
    print("Current segments count: ", len(features))
