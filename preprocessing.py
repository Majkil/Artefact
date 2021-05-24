from utilities import *
import GetTranscription
from energy_split import *
from m_dictionary import *


def split_into_segments(audio, hop_length, frame_length, sr, min_voiced_duration_ms=10, energy_threshold=0.05):
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
                
                if len(phone_array)==0:
                    phone_array.append(char)
                elif char == 'ː':
                    phone_array[-1] = phone_array[-1] + char
                elif char == 'ɪ' and (phone_array[-1] == 'e' or phone_array[-1] == 'a' or phone_array[-1] == 'ɔ'):
                    phone_array[-1] = phone_array[-1] + char
                elif char == 'ʊ' and (phone_array[-1] == 'o' or phone_array[-1] == 'a'):
                    phone_array[-1] = phone_array[-1] + char
                elif char == 'ʒ' and (phone_array[-1] == 'd'):
                    phone_array[-1] = phone_array[-1] + char
                elif char == 'ʃ' and (phone_array[-1] == 't'):
                    phone_array[-1] = phone_array[-1] + char
                elif char == 'ə' and (phone_array[-1] == 'e' or phone_array[-1] == 'ɪ' or phone_array[-1] == 'ʊ'):
                    phone_array[-1] = phone_array[-1] + char
                else:
                    phone_array.append(char)
    return phone_array


def all_phoneme_Sections_in_clip(audio, segments, sr, frame_length, hop_length, min_duration=700):
    all_bits = []
    for segment in segments:
        starting = segment[0]
        segment_boundaries = Split2(audio[starting:segment[1]], hop_length=hop_length,
                                    frame_length=frame_length, sr=sr, min_duration=min_duration)
        for bit in segment_boundaries:
            x1 = starting + bit[0]
            x2 = starting + bit[1]
            b = (x1, x2)
            all_bits.append(b)
    return all_bits


def all_phoneme_Sections_in_clip2(audio, segments, hop_length, sr , min_duration):
    all_bits = []
    for segment in segments:
        starting = segment[0]
        ending = segment[1]
        segment_boundaries = Split3(
            audio[starting:ending], sr = sr, hop_length=hop_length, min_duration=min_duration )
        for bit in segment_boundaries:
            x1 = starting + bit[0]
            x2 = starting + bit[1]
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


def process_clip(audio_path, expected_phonemes):
    sr = 12000
    hl = int(sr / 100)
    fl = hl * 2

    audio = load_clip(audio_path, sr)
    segments = split_into_segments(audio, hl, fl, sr)
    phoneme_bits = all_phoneme_Sections_in_clip(audio, segments, sr=sr, frame_length=fl, hop_length=hl,
                                                min_duration=hl * 8)
    counter = 1
    threshold = 0.04

    while len(phoneme_bits) != expected_phonemes and counter <= 12:
        # hl = int(sr/100)*counter

        if len(phoneme_bits) > expected_phonemes:
            if sr < 18000:
                sr += 1000
                audio = load_clip(audio_path, sr)
            threshold += 0.005
            if hl > int(sr / 100):
                hl = int(sr / 50)
            elif hl > int(sr / 200):
                hl = int(sr / 100)
            elif hl > int(sr / 250):
                hl = int(sr / 200)
        else:
            threshold -= 0.005
            if hl < int(sr / 100):
                hl = int(sr / 200)
            elif hl < int(sr / 200):
                hl = int(sr / 250)
        fl = hl * 2
        segments = split_into_segments(
            audio, hl, fl, sr, energy_threshold=threshold)
        phoneme_bits = all_phoneme_Sections_in_clip(
            audio, segments, sr, fl, hl, )
        counter += 1
    if len(phoneme_bits) == expected_phonemes:
        return segments, phoneme_bits, sr
    else:
        return segments, [], sr


def process_clip2(audio_path, expected_phonemes):
    sr = 22000
    hl = int(sr / 500)
    fl = hl * 2
    threshold = 0.025

    audio = load_clip(audio_path, sr)
    segments = split_into_segments(
        audio, hl, fl, sr, energy_threshold=threshold, min_voiced_duration_ms=50)
    trimmed = clip_from_segments(audio, segments)
    voiced_time=  len(trimmed) /sr
    min_duration = math.ceil(voiced_time/expected_phonemes*1000)
    phoneme_bits = all_phoneme_Sections_in_clip2( audio, segments, hop_length=hl ,sr = sr, min_duration=min_duration)
        
    counter = 1

    while len(phoneme_bits) != expected_phonemes and counter <= 12:
        
        if len(phoneme_bits) > expected_phonemes:
            if sr < 25000:
                sr += 1000
                audio = load_clip(audio_path, sr)
            threshold += 0.005
            if hl > int(sr / 100):
                hl = int(sr / 50)
            elif hl > int(sr / 200):
                hl = int(sr / 100)
            elif hl > int(sr / 250):
                hl = int(sr / 200)
        else:
            threshold -= 0.005
            if hl < int(sr / 100):
                hl = int(sr / 200)
            elif hl < int(sr / 200):
                hl = int(sr / 250)
        fl = hl * 2
        segments = split_into_segments(
            audio, hl, fl, sr, energy_threshold=threshold, min_voiced_duration_ms=min_duration)
        phoneme_bits = all_phoneme_Sections_in_clip2(
            audio, segments, hop_length=hl,sr=sr,min_duration= min_duration)
        counter += 1
    if len(phoneme_bits) == expected_phonemes:
        return segments, phoneme_bits, sr
    else:
        return segments, [], sr
