import os
import numpy as np

import audio_utilities

data = {}
print("Loading audio...")
fft_size = 2048
sample_rate = 8000
max_len = 80
mel = False
for root, dirs, files in os.walk("./audio"):
    if len(files) == 0:
        continue
    group = os.path.split(root)[-1]
    if group == "audio":
        continue
    data[group] = []
    for f in sorted(files):
        if f.endswith(".wav"):
            input_signal = audio_utilities.get_signal(
                os.path.join(root, f), expected_fs=sample_rate)
            # Hopsamp is the number of samples that the analysis window is shifted after
            # computing the FFT. For example, if the sample rate is 44100 Hz and hopsamp is
            # 256, then there will be approximately 44100/256 = 172 FFTs computed per second
            # and thus 172 spectral slices (i.e., columns) per second in the spectrogram.
            hopsamp = fft_size // 8

            # Compute the Short-Time Fourier Transform (STFT) from the audio file. This is a 2-dim Numpy array with
            # time_slices rows and frequency_bins columns. Thus, you will need to take the
            # transpose of this matrix to get the usual STFT which has frequency bins as rows
            # and time slices as columns.
            stft_full = audio_utilities.stft_for_reconstruction(input_signal,
                                                                fft_size, hopsamp)

            # If maximum length exceeds mfcc lengths then pad the remaining ones
            if (max_len > stft_full.shape[0]):
                pad_width = max_len - stft_full.shape[0]
                stft_pad = np.pad(stft_full, pad_width=(
                    (0, pad_width), (0, 0)), mode='constant')
            # Else cutoff the remaining parts
            else:
                stft_pad = stft_full[:max_len, :]
            # Note that the STFT is complex-valued. Therefore, to get the (magnitude)
            # spectrogram, we need to take the absolute value.
            stft_mag = abs(stft_pad)**2.0
            # Note that `stft_mag` only contains the magnitudes and so we have lost the
            # phase information.
            scale = 1.0 / np.amax(stft_mag)
            # print('Maximum value in the magnitude spectrogram: ', 1/scale)
            # Rescale to put all values in the range [0, 1].
            stft_mag *= scale

            if mel:
                min_freq_hz = 70
                max_freq_hz = 8000
                mel_bin_count = 200

                linear_bin_count = 1 + fft_size//2
                filterbank = audio_utilities.make_mel_filterbank(min_freq_hz, max_freq_hz, mel_bin_count,
                                                                 linear_bin_count, sample_rate)
                mel_spectrogram = np.dot(filterbank, stft_mag.T)
                inverted_mel_to_linear_freq_spectrogram = np.dot(
                    filterbank.T, mel_spectrogram)
                strf_mag = inverted_mel_to_linear_freq_spectrogram.T
                print(strf_mag.shape)
            data[group].append((stft_mag, scale))

    if not os.path.exists("cache"):
        os.mkdir("cache")
    np.save(os.path.join("cache", group + '.npy'), data[group])
