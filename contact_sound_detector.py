import sys
from aubio import source
from aubio import pitch as get_pitch

MIN_SWING_BUFF = 1.0  # maybe 1.0 for wall and 1.25 for rallying
MIN_CONTACT_PITCH = 80
SAMPLE_PITCH_RATE = 0.25
SWING_WINDOW = 0.75


def detect_contacts(filename):
    print(filename)
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

    total_frames = 0
    max_pitch = 0
    last_pitch = 0
    last_contact = 0
    st_swing_win = -1
    max_swing_time = -1
    max_swing_pitch = -1
    contacts = []
    while True:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        # pitch = int(round(pitch))
        timestamp = total_frames / float(samplerate)
        # confidence = pitch_o.get_confidence()
        # if confidence < 0.1:
        #     pitch = 0
        # print("%f %f" % (timestamp, pitch))
        total_frames += read

        # no swing detected
        if st_swing_win == -1:
            if pitch > max_pitch:
                max_pitch = pitch
            # get max pitch in sample rate
            if timestamp - last_pitch >= SAMPLE_PITCH_RATE:
                # swing window detected
                if max_pitch > MIN_CONTACT_PITCH and timestamp >= last_contact + MIN_SWING_BUFF:
                    last_contact = timestamp
                    st_swing_win = timestamp
                    max_swing_time = timestamp
                    max_swing_pitch = max_pitch
                    print('poss contact {} {}'.format(
                        st_swing_win, max_swing_pitch))
                last_pitch = timestamp
                max_pitch = 0
        else:
            if timestamp - st_swing_win < SWING_WINDOW and pitch > max_swing_pitch:
                max_swing_time = timestamp
                max_swing_pitch = pitch
                print('updating max {} {}'.format(
                    st_swing_win, max_swing_pitch))
            # get largest pitch in swing window
            elif timestamp - st_swing_win >= SWING_WINDOW:
                contacts.append(round(max_swing_time, 4))
                print('contact {} {}'.format(max_swing_time, max_swing_pitch))
                st_swing_win = -1
                max_swing_time = -1
                max_swing_pitch = -1
                last_contact = max_swing_time

        if read < hop_s:
            break

    return contacts


if __name__ == "__main__":
    contacts = detect_contacts(sys.argv[1])
    print(contacts)
