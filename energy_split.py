from numpy import lib
from utilities import *
import math


def split_segments(audio, hop_length, frame_length, sr, min_duration=10, energy_threshold=0.05):
    ##
    # @brief Returns voiced sections depending on set energy theshold
    #
    # Keyword argument:
    # @param audio -- librosa audio sequence
    # @param hop_length -- hop length in frames
    # @param sr -- audio sampling rate
    # @param min_duration -- minimum section duration in milli seconds
    # @param energy_threshold -- minimum energy to be considered non silent in percentage

    min_duration = (sr / 1000) * min_duration
    f = librosa.feature.rms(np.abs(audio), hop_length=hop_length, frame_length=frame_length)[0]
    start, end = 0, 0
    voiced = []

    n = normalize(f)

    above_t = np.where(n >= energy_threshold)[0]
    print(np.where(n <= energy_threshold)[0])
    for x in range(len(above_t) - 1):
        # if next frame in next in sequence
        if above_t[x] + 1 == above_t[x + 1] and start == 0:
            start = above_t[x]
        # check upto 3 frames away for sequence continuation
        elif above_t[x + 1] > above_t[x] +2 and start != 0:
            end = above_t[x]
            voiced.append([start, end])
            start, end = 0, 0

    trimmed = []
    for x in voiced:
        if len(signal.argrelmin(audio[x[0]:x[1]])[0])>1:
            trimmed.append(x)

    if len(trimmed) == 0:
        trimmed = [[0, len(f)]]
    return np.array(trimmed) * hop_length


def Split2(audio, hop_length, frame_length, sr, min_duration=700):
    if len(audio) <= 2 * min_duration:
        return [[0, len(audio)]]
    # audio = librosa.load(audio,sr=sr)
    e = librosa.pcen(np.abs(audio), sr=sr, hop_length=hop_length)
    d = librosa.feature.delta(e)
    start, end = 0, 0
    boundaries = []
    positive = True
    boundaries.append(0)
    # diff = 0.8
    # calulate delta Y  = m
    # dy = sqrt(m^2)
    # take largest dy as boundary
    # take lowest dy as turning point1
    # take next value close to boundary value as closing value
    for c in range(len(d)):
        if d[c] < d[c - 1] and d[c] < d[c - 2] and d[c] < d[c + 1] and d[c] < d[c + 2] and boundaries[
            -1] + min_duration <= c:
            boundaries.append(c)
    x = []
    if len(d) - boundaries[-1] < min_duration:
        boundaries[-1] = len(d)
    else:
        boundaries.append(len(d))
    if len(boundaries) > 1 and boundaries[-1] == boundaries[-2]:
        boundaries.pop()

    segments = []
    for b in range(len(boundaries)):
        if b < len(boundaries) - 1:
            segments.append([boundaries[b], boundaries[b + 1]])
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


def Split3(audio, hop_length, sr, min_duration=300):
    # get RMS energy and apply preemphasis filter
    sec_energy = librosa.feature.rms(np.abs(audio), hop_length=hop_length)[0]
    sec_energy = librosa.effects.preemphasis(sec_energy)
    mins = signal.argrelextrema(sec_energy, np.less)[0]
    if not mins.any():
        mins = np.append(mins, 0)
        mins = np.append(mins, len(sec_energy))
    tups = []
    maxs = signal.argrelextrema(sec_energy, np.greater)[0]
    # if too short to have at least 2 subsections
    if (len(mins) > 0 or len(maxs) > 0) and len(audio) / sr < 2 * min_duration / 1000:
        return [(0, len(audio))]
    if len(mins) == 1 and mins[0] < len(audio) / hop_length / 3 and len(maxs) == 1 and len(audio) / hop_length / 3 < \
            maxs[0] < len(audio) / hop_length / 3 * 2:
        tups.append((0, maxs[0]))
        tups.append((maxs[0], math.ceil(len(audio) / hop_length)))
        return np.array(tups) * hop_length
    # mins = [*mins,*maxs]
    mins.tolist().append(math.ceil(len(audio) / hop_length))
    mins.sort()

    # not enough usable peaks or valleys detected
    if len(mins) <= 1:
        if len(maxs) == 2:
            tups.append((maxs[0], maxs[1]))
        elif mins[0] < 1 and (mins[0] * hop_length) / sr < min_duration / 1000:
            tups.append((0, mins[0]))
        elif mins[0] < 1 and ((mins[0] * hop_length) - len(audio)) / sr < min_duration / 1000:
            tups.append((mins[0], len(audio)))
        elif len(maxs) == 1 and maxs[0] != 0 and (maxs[0] * hop_length) / sr < min_duration / 1000:
            tups.append((0, maxs[0]))
        elif len(maxs) > 2:
            mins = maxs
        else:
            return [()]
    # create tuples for subsections
    for i in range(len(mins) - 1):
        # distance between next two valleys less than min_duration
        if (mins[i + 1] - mins[i]) * hop_length / sr < min_duration / 1000:
            # if shorter than min and this is the last valley, add it to the last segment
            if i == len(mins) - 1:
                tups[-1] = (tups[-1][0], mins[i])
            # if not first segment and previous segment is shorter than minimum duration, add it to the last segment
            if len(tups) >= 1 and (tups[-1][1] - tups[-1][0]) * hop_length / sr < min_duration / 1000:
                tups[-1] = (tups[-1][0], mins[i])
            # if previous segment is long enough start a new segment
            elif len(tups) >= 1:
                tups.append((tups[-1][1], mins[i]))
            # else this is the first segment
            else:
                tups.append((0, mins[i]))

        # if distance between next 2 valleys is the >= min duration
        elif (mins[i + 1] - mins[i]) * hop_length / sr >= min_duration / 1000:
            # if previous segment is too short append it
            if len(tups) > 0 and tups[-1][1] - tups[-1][0] * hop_length / sr > min_duration / 1000:
                tups[-1] = (tups[-1][0], mins[i])
            # else create a new segment with next 2 valleys
            else:
                tups.append((mins[i], mins[i + 1]))
        # this code should never be reached
        else:
            try:
                tups.append((tups[-1][1], mins[i]))
            except:
                if mins[i] != 0 and mins[i] * hop_length / sr < min_duration / 1000:
                    tups.append((0, mins[i + 1]))
    # handle final value
    if (mins[-1] - mins[-2]) * hop_length / sr > min_duration:
        tups.append((mins[-2], mins[-1]))
    if (tups[-1][1] - tups[-1][0]) * hop_length / sr < min_duration / 1000 or (
            mins[-1] - mins[-2]) * hop_length / sr < min_duration / 1000:
        tups[-1] = (tups[-1][0], mins[-1])
    else:
        tups.append((tups[-1][1], mins[-1]))

    return np.array(tups) * hop_length


def Split4(segment, sr, expected_phoneme_count, min_duration=80):
    # get RMS energy and apply preemphasis filter
    hop_length = int(sr / 200)
    frame_length = int(hop_length * 2.5)
    mins = [0]
    sec_energy = librosa.feature.rms(np.abs(segment), hop_length=hop_length, frame_length=frame_length)[0]
    sec_energy = librosa.effects.preemphasis(sec_energy)
    mins.extend(signal.argrelextrema(sec_energy, np.less)[0] * hop_length)

    while len(mins) > expected_phoneme_count * 2.5:
        hop_length = int(hop_length * 1.05)
        frame_length = int(hop_length * 2.5)
        sec_energy = librosa.feature.rms(np.abs(segment), hop_length=hop_length, frame_length=frame_length)[0]
        sec_energy = librosa.effects.preemphasis(sec_energy)
        mins = signal.argrelextrema(sec_energy, np.less)[0]
        temp_mins = [0]
        means = []
        for x in range(0, len(mins) - 1):
            means.append(np.mean(segment[mins[x]:mins[x + 1]]))
            # print(np.mean(segment[mins[x]:mins[x+1]]))
        means = normalize(means)
        for x in range(0, len(means) - 1):
            diff = means[x + 1] - means[x]
            if diff >= 0.06:
                temp_mins.append(mins[x])
        mins = temp_mins

    mins = np.array(mins)
    if not mins.any():
        mins = np.append(mins, 0)
        mins = np.append(mins, len(sec_energy))
    tups = []
    maxs = signal.argrelextrema(sec_energy, np.greater)[0]
    # if too short to have at least 2 subsections
    if (len(mins) > 0 or len(maxs) > 0) and len(segment) / sr < 2 * min_duration / 1000:
        return [(0, len(segment))]
    if len(mins) == 1 and mins[0] < len(segment) / hop_length / 3 and len(maxs) == 1 and len(segment) / hop_length / 3 < \
            maxs[0] < len(audio) / hop_length / 3 * 2:
        tups.append((0, maxs[0]))
        tups.append((maxs[0], math.ceil(len(segment) / hop_length)))
        return np.array(tups) * hop_length
    # mins = [*mins,*maxs]
    mins.tolist().append(math.ceil(len(segment) / hop_length))
    mins.sort()

    # not enough usable peaks or valleys detected
    if len(mins) <= 1:
        if len(maxs) == 2:
            tups.append((maxs[0], maxs[1]))
        elif mins[0] < 1 and (mins[0] * hop_length) / sr < min_duration / 1000:
            tups.append((0, mins[0]))
        elif mins[0] < 1 and ((mins[0] * hop_length) - len(segment)) / sr < min_duration / 1000:
            tups.append((mins[0], len(segment)))
        elif len(maxs) == 1 and maxs[0] != 0 and (maxs[0] * hop_length) / sr < min_duration / 1000:
            tups.append((0, maxs[0]))
        elif len(maxs) > 2:
            mins = maxs
        else:
            return [()]
    # create tuples for subsections
    for i in range(len(mins) - 1):
        # distance between next two valleys less than min_duration
        if (mins[i + 1] - mins[i]) * hop_length / sr < min_duration / 1000:
            # if shorter than min and this is the last valley, add it to the last segment
            if i == len(mins) - 1:
                tups[-1] = (tups[-1][0], mins[i])
            # if not first segment and previous segment is shorter than minimum duration, add it to the last segment
            if len(tups) >= 1 and (tups[-1][1] - tups[-1][0]) * hop_length / sr < min_duration / 1000:
                tups[-1] = (tups[-1][0], mins[i])
            # if previous segment is long enough start a new segment
            elif len(tups) >= 1:
                tups.append((tups[-1][1], mins[i]))
            # else this is the first segment
            else:
                tups.append((0, mins[i]))

        # if distance between next 2 valleys is the >= min duration
        elif (mins[i + 1] - mins[i]) * hop_length / sr >= min_duration / 1000:
            # if previous segment is too short append it
            if len(tups) > 0 and tups[-1][1] - tups[-1][0] * hop_length / sr > min_duration / 1000:
                tups[-1] = (tups[-1][0], mins[i])
            # else create a new segment with next 2 valleys
            else:
                tups.append((mins[i], mins[i + 1]))
        # this code should never be reached
        else:
            try:
                tups.append((tups[-1][1], mins[i]))
            except:
                if mins[i] != 0 and mins[i] * hop_length / sr < min_duration / 1000:
                    tups.append((0, mins[i + 1]))
    # handle final value
    if (mins[-1] - mins[-2]) * hop_length / sr > min_duration:
        tups.append((mins[-2], mins[-1]))
    if (tups[-1][1] - tups[-1][0]) * hop_length / sr < min_duration / 1000 or (
            mins[-1] - mins[-2]) * hop_length / sr < min_duration / 1000:
        tups[-1] = (tups[-1][0], mins[-1])
    else:
        tups.append((tups[-1][1], mins[-1]))

    return np.array(tups) * hop_length



def split_segment_depricated(audio, hop_length, frame_length, sr, min_duration=10, energy_threshold=0.05):
    min_duration = (sr / 1000) * min_duration
    f = librosa.feature.rms(audio, hop_length=hop_length, frame_length=frame_length)[0]
    start, end = 0, 0
    voiced = []

    n = normalize(f)
    for x in range(len(n)):
        if n[x] > energy_threshold:
            if start == 0:
                start = x
                print(x, len(n))
            if start != 0 and x < len(n) - 2:
                if n[x + 1] < energy_threshold and start != x:
                    end = x
                    voiced.append([start, end])
                    start, end = 0, 0

    trimmed = []

    for x in voiced:
        diff = x[1] - x[0]
        if (diff * hop_length) >= min_duration:
            trimmed.append([x[0], x[1]])
    if len(trimmed) == 0:
        trimmed = [[0, len(audio)]]
    return np.array(trimmed) * hop_length

