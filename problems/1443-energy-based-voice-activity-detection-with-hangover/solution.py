import numpy as np


def frame_rms(x, frame_len, hop):
    frame_count = (x.shape[0] - frame_len) // hop + 1
    frame_results = []
    for i in range(frame_count):
        frame_start = i * hop
        frame = x[frame_start: frame_start+frame_len]
        frame_results.append(np.sqrt(np.mean(np.square(frame))))
    return np.asarray(frame_results)

def apply_hangover(flags, hangover_frames):
    shifted_flags = np.asarray(flags).copy()
    for i in range(min(hangover_frames, len(flags))):
        shifted_flag = np.pad(flags[:len(flags) - i - 1], (i + 1,0), "constant", constant_values=False).astype(bool)
        shifted_flags |= shifted_flag
    return shifted_flags

def segments_from_flags(flags, hop, sr, min_speech_s):
    # TODO: runs of True -> (start_s, end_s) rounded to 4 decimals; drop runs shorter than min_speech_s
    start_frame = -1
    segments = []
    for i in range(len(flags)):
        if flags[i] and start_frame == -1:
            # Continue or start a new window
            start_frame = i
        elif not flags[i] and start_frame != -1:
            # A previously open window is now closed
            start_s = start_frame * hop / sr
            end_s = i * hop / sr
            start_frame = -1

            if end_s - start_s >= min_speech_s:
                segments.append((float(start_s), float(end_s)))

    if start_frame != -1:
        # Add last entry if still open
        start_s = start_frame * hop / sr
        end_s = len(flags) * hop / sr
        start_frame = -1

        if end_s - start_s >= min_speech_s:
            segments.append((float(start_s), float(end_s)))

    return segments

def vad(x, sr, frame_len, hop, threshold, hangover_frames, min_speech_s):
    rms = frame_rms(x, frame_len, hop)
    flags = rms > threshold
    hangover_frames = apply_hangover(flags, hangover_frames)
    segments = segments_from_flags(hangover_frames, hop, sr, min_speech_s)
    return segments
