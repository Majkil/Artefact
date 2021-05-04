from utilities import *


def energies(audio, hop_length, frame_length):

    energy = normalize(librosa.feature.rms(
        audio, frame_length=frame_length, hop_length=hop_length)[0])
    delta = librosa.feature.delta(energy)
    return librosa.util.stack([energy, delta], axis=1)


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
