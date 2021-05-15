from utilities import *
import GetTranscription
from energy_split import *
from m_dictionary import *


def split_into_segments(audio, hop_length, frame_length, sr, min_voiced_duration_ms=500, energy_threshold=0.05):
    segments = Split(audio, hop_length, frame_length, sr,
                     min_voiced_duration_ms, energy_threshold)
    return segments


def load_clip(clip_address, sr):
    audio, sr = librosa.load(clip_address, sr=sr)
    return audio


def load_clip_transcription(clip_address):
    transcription = GetTranscription.get_file_transcript(clip_address)
    return transcription


def all_phones_to_array(transcription):
    clip_phones = get_phonemes_for_sentence(sentence=transcription)
    phone_array = []
    for word in clip_phones:
        if len(word) == word.count(' '):
            phone_array.append('XXXXXX')
        if len(word) != word.count(' '):
            # print(len(word))
            # print(word)
            for char in word:
                if char == 'ː':
                    phone_array[-1] = phone_array[-1]+char
                else:
                    phone_array.append(char)
    return phone_array


def all_phoneme_Sections_in_clip(audio, segments, sr, frame_length, hop_length, min_duration):
    all_bits = []
    for segment in segments:
        starting = segment[0]
        segment_boundaries = Split2(audio[starting:segment[1]], hop_length=hop_length,
                                    frame_length=frame_length, sr=sr, min_duration=hop_length*10)
        for bit in segment_boundaries:
            x1 = starting+bit[0]
            x2 = starting+bit[1]
            b = (x1, x2)
            all_bits.append(b)
    return all_bits


def clip_from_segments(audio, segments):
    voiced = []
    for x in segments:
        for i in audio[x[0]:x[1]]:
            voiced.insert(len(voiced), i)
    voiced = np.array(voiced)
    print(voiced.shape)
    return voiced
