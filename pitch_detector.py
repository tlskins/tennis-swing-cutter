import sys
from aubio import source
from aubio import pitch as get_pitch
import numpy as num
import datetime

MIN_SWING_BUFF = 1.5  # maybe 1.0 for wall and 1.25 for rallying
MIN_CONTACT_PITCH = 50
SAMPLE_PITCH_RATE = 0.25
SWING_WINDOW = 0.75
FPS = 30

def detect_pitches(filename):
    print("getting sound frames... ", filename)
    t1 = datetime.datetime.now()
    downsample = 1
    samplerate = 44100 // downsample
    win_s = 4096 // downsample  # fft size
    hop_s = 512 // downsample  # hop size

    s = source(filename, samplerate, hop_s)
    samplerate = s.samplerate

    tolerance = 0.8

    pitch_o = get_pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)

    total_audio_frames = 0
    frames = []
    max_pitch = 0
    last_read = 0
    while True:
        samples, read = s()
        total_audio_frames += read
        pitch = pitch_o(samples)[0]
        pitch = int(round(pitch, 2))
        max_pitch = pitch if pitch > max_pitch else max_pitch
        timestamp = total_audio_frames / float(samplerate)
        frame_num = round(timestamp * FPS)
        
        if frame_num - last_read >= 1.0:
            if pitch >= MIN_CONTACT_PITCH:
                frames.append(frame_num)
            max_pitch = 0
            last_read = frame_num

        if read < hop_s:
            break
    print('calc sound frames: {} seconds'.format((datetime.datetime.now()-t1).seconds))


    return frames


if __name__ == "__main__":
    pitches = detect_pitches(sys.argv[1])
    print(pitches)
