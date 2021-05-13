from utilities import *
import math

##
# @brief Returns voiced sections depending on set energy theshold
#
# Keyword argument:
# @param audio -- librosa audio sequence
# @param hop_length -- hop length in frames
# @param sr -- audio sampling rate
# @param min_duration -- minimum section duration in milli seconds
# @param energy_threshold -- minimum energy to be considered non silent in percentage


def Split(audio, hop_length, frame_length, sr, min_duration=10,  energy_threshold=0.05):
    min_duration = (sr/1000)*min_duration
    f = librosa.feature.rms(audio, hop_length=hop_length,frame_length=frame_length)[0]
    start, end = 0, 0
    voiced = []
    n = normalize(f)
    for x in range(len(n)):
        if n[x] > energy_threshold:
            if start == 0:
                start = x
            if start != 0 and x < len(n):
                if n[x+1] < energy_threshold and start != x:
                    end = x
                    voiced.append([start, end])
                    start, end = 0, 0

    trimmed = []

    for x in voiced:
        diff = x[1]-x[0]
        if diff*hop_length >= min_duration:
            trimmed.append([x[0]*hop_length,x[1]*hop_length])

    return trimmed


def Split2(audio, hop_length, frame_length, sr,  min_duration=700):

    if len(audio) <= 2*min_duration:
        return [[0, len(audio)]]
    e = librosa.pcen(audio, sr=sr)
    d = librosa.feature.delta(e)
    start, end = 0, 0
    boundaries = []
    positive = True
    boundaries.append(0)
    #diff = 0.8
    # calulate delta Y  = m
    # dy = sqrt(m^2)
    # take largest dy as boundary
    # take lowest dy as turning point1
    # take next value close to boundary value as closing value
    for c in range(len(d)):
        if d[c] < d[c-1] and d[c] < d[c-2] and d[c] < d[c+1] and d[c] < d[c+2] and boundaries[-1]+min_duration <= c:
            boundaries.append(c)
    x = []
    if len(d) - boundaries[-1] < min_duration:
        boundaries[-1] = len(d)
    else:
        boundaries.append(len(d))
    if len(boundaries)>1 and boundaries[-1] == boundaries[-2]:
        boundaries.pop()
    
    segments = []
    for b in range(len(boundaries)):
        if b < len(boundaries)-1:
            segments.append([boundaries[b],boundaries[b+1]])
    return segments

    # gradients = []
    # for y in range(len(d)):
    #     if y < len(d)-1:
    #         m = d[y]-d[y+1]
    #         dy = math.sqrt(m**2)
    #         gradients.append(dy)
    # highest_gradient = 0

    # for i in range(len(gradients)):
    #     if i < len(gradients)-1 and boundaries[-1]+min_duration = i:
    #         if gradients[i] > gradients[i+1] and gradients[i] > highest_gradient:
    #             highest_gradient = gradients[i]
    #             boundaries.append(i)
    #         if gradients[i] > gradients[i+1] and gradients[i] < highest_gradient:
    #             boundaries.append(i)
    #             highest_gradien = 0
    return boundaries


def split_by_energy(audio, hop_length, frame_length):
    frames = len(audio)
    # root mean squared energy
    energy = librosa.feature.rms(
        audio, frame_length=frame_length, hop_length=hop_length)[0]
    # rate of change of energy
    delta_energy = librosa.feature.delta(energy)
    # rate of change of change of energy
    delta_energy2 = librosa.feature.delta(delta_energy)

    scaled_audio = normalize(audio)
    rms_norm = normalize(energy)
    s_d_energy = normalize(delta_energy)
    s_d_2_energy = normalize(delta_energy2)

    audio_range = np.max(scaled_audio) - np.min(scaled_audio)
    print(audio_range)
    mean = np.mean(scaled_audio)

    #print("scaled delta energy less than 0.5 ", np.count_nonzero( s_d_energy<0.8))
    #print("scaled audio less than 0.5 ", np.count_nonzero(0.45 >scaled_audio or scaled_audio> 0.55))

    out = []
    # blue audio
    # yellow energy
    # green de1
    # red de2
    for x in range(frames):
        if scaled_audio[x] > (mean+0.01) or scaled_audio[x] < (mean-0.01):

            if s_d_energy[x] < 0.8:
                out.append(x)
    return out
