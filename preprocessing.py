from m_dictionary import *
from energy_split import *
import GetTranscription
from utilities import *



def split_into_segments(audio, hop_length, frame_length, sr, min_voiced_duration_ms=10, energy_threshold=0.05):
    segments = split_segments(audio, hop_length, frame_length, sr,
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

                if len(phone_array) == 0:
                    phone_array.append(char)
                elif char == 'ː':
                    phone_array[-1] = phone_array[-1] + char
                elif char == 'ɪ' and (phone_array[-1] == 'e' or phone_array[-1] == 'a' or phone_array[-1] == 'ɔ'):
                    phone_array[-1] = phone_array[-1] + char
                elif char == 'ʊ' and (phone_array[-1] == 'o' or phone_array[-1] == 'a' or phone_array[-1] == 'ə'):
                    phone_array[-1] = phone_array[-1] + char
                elif char == 'ʒ' and (phone_array[-1] == 'd'):
                    phone_array[-1] = phone_array[-1] + char
                elif char == 'ʃ' and (phone_array[-1] == 't'):
                    phone_array[-1] = phone_array[-1] + char
                elif char == 'ə' and (phone_array[-1] == 'e' or phone_array[-1] == 'ɪ' or phone_array[-1] == 'ʊ' or phone_array[-1] == 'ð'):
                    phone_array[-1] = phone_array[-1] + char
                elif char == 'ɐ' and (phone_array[-1] == 'ð'):
                    phone_array[-1] = phone_array[-1] + char
                else:
                    phone_array.append(char)
    return phone_array

def auto_process(clip_address):
    transcript = load_clip_transcription(clip_address)
    phonemes = all_phones_to_array(transcript)

    sr = 22000
    hl = int(sr / 200)
    fl = int(hl * 2)
    audio, sr = librosa.load(clip_address, sr)
    segments = split_segments(audio, hl, fl, sr)
    #print("segmented words:", len(segments))
    all_bits = []
    for seg in segments:
        s, e = seg[0], seg[1]
        seg_audio = audio[s:e]
        boundaries = phonme_boundaries(seg_audio, sr)
        bits = boundaries_to_segments(boundaries)
        for b in range(len(bits)):
            all_bits.append(audio[s + bits[b][0]:s + bits[b][1]])
    #print(transcript, "\n || word count: ", len(transcript.split(" ")), "|| phoneme count:", len(phonemes))
    #print("\n ||segmented words:", len(segments), " || detected phonemes:", len(all_bits))
    return len(transcript.split(" ")), len(segments), len(phonemes), len(all_bits) , all_bits

#Depricated
def all_phoneme_Sections_in_clip(audio, segments, sr, frame_length, hop_length, min_duration=700):
#Depricated
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

#Depricated
def all_phoneme_Sections_in_clip2(audio, segments, hop_length, sr, min_duration):
#Depricated
    all_bits = []
    for segment in segments:
        starting = segment[0]
        ending = segment[1]
        segment_boundaries = Split3(
            audio[starting:ending], sr=sr, hop_length=hop_length, min_duration=min_duration)
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
    # print(voiced.shape)
    return voiced


def process_clip2(audio_path, expected_phonemes):
    sr = 22000
    hl = int(sr / 500)
    fl = hl * 2
    threshold = 0.025

    audio = load_clip(audio_path, sr)
    segments = split_into_segments(
        audio, hl, fl, sr, energy_threshold=threshold, min_voiced_duration_ms=50)
    trimmed = clip_from_segments(audio, segments)
    voiced_time = len(trimmed) / sr
    min_duration = math.ceil(voiced_time/expected_phonemes*1000)
    phoneme_bits = all_phoneme_Sections_in_clip2(
        audio, segments, hop_length=hl, sr=sr, min_duration=min_duration)

    counter = 1
# region
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
            audio, segments, hop_length=hl, sr=sr, min_duration=min_duration)
        counter += 1
    if len(phoneme_bits) == expected_phonemes:
        return segments, phoneme_bits, sr
    else:
        return segments, [], sr

    # while len(phoneme_bits) != expected_phonemes and counter <= 12:
    #     direction = 3
    #     delta = 0.1

    #     if len(phoneme_bits) > expected_phonemes:
    #         if direction == 1:
    #             delta = delta*1.5
    #         else:
    #             delta = delta*0.75

    #         if sr < 25000:
    #             sr += 1000
    #             audio = load_clip(audio_path, sr)
    #         threshold = threshold + delta
    #         if hl > int(sr / 100):
    #             hl = int(sr / 50)
    #         elif hl > int(sr / 200):
    #             hl = int(sr / 100)
    #         elif hl > int(sr / 250):
    #             hl = int(sr / 200)

    #         direction = 1
    #     else:
    #         if direction == 2:
    #             delta = delta*1.5
    #         else:
    #             delta = delta*0.75

    #         threshold = threshold * delta
    #         if hl < int(sr / 100):
    #             hl = int(sr / 200)
    #         elif hl < int(sr / 200):
    #             hl = int(sr / 250)

    #         direction = 2
    #     fl = hl * 2
    #     segments = split_into_segments(
    #         audio, hl, fl, sr, energy_threshold=threshold, min_voiced_duration_ms=min_duration)
    #     phoneme_bits = all_phoneme_Sections_in_clip2(
    #         audio, segments, hop_length=hl,sr=sr,min_duration= min_duration)
    #     counter += 1
    # if len(phoneme_bits) == expected_phonemes:
    #     return segments, phoneme_bits, sr
    # else:
    #     return segments, [], sr
# endregion


def process_clip_with_fb(clip_address):
    sr = 22000
    hl = int(sr / 200)
    fl = hl*2
    audio = load_clip(clip_address, sr)
    segments = split_into_segments(
        audio, hop_length=hl, frame_length=fl, sr=sr, min_voiced_duration_ms=200, energy_threshold=0.05)
    transcription = GetTranscription.get_file_transcript(clip_address)
    return_bits = []
    return_labels = []
    for segment in segments:
        s = segment[0]
        e = segment[1]
        transcription_pred = transcribe_audio_fb(audio=audio[s:e])

        if len(transcription_pred) > 0 and transcription.find(transcription_pred) >= 0:
            min_duration = math.ceil(
                ((segment[1]-segment[0])/sr)/len(transcription_pred)*1000)

            seg_data = audio[segment[0]:segment[1]]
            graphemes = all_phones_to_array(transcription_pred)
            if 'XXXXXX' not in graphemes:
                phoneme_bits = Split4(seg_data, sr=sr, expected_phoneme_count=len(
                    graphemes), min_duration=min_duration)
                #phoneme_bits = all_phoneme_Sections_in_clip2(                audio=audio, segments=[segment], sr=sr, hop_length=hl, min_duration=min_duration)
                while len(graphemes) < len(phoneme_bits):
                    min_duration += 5
                    phoneme_bits = Split4(seg_data, sr=sr, expected_phoneme_count=len(
                        graphemes), min_duration=min_duration)
                if len(graphemes) == len(phoneme_bits) and len(graphemes) != 0:
                    return_labels.extend(graphemes)
                    for b in phoneme_bits:
                        if len(b)> 0:
                            return_bits.append(audio[s+b[0]:s+b[1]])

    return return_bits, return_labels

#Depricated
def process_clip_deprecated(audio_path, expected_phonemes):
#Depricated
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
