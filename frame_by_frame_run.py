# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:37:43 2019

@author: a-kojima

add WPE

"""
from beamformer import frame_by_frame_mvdr as fbf_mvdr
from beamformer import util

import numpy as np
import soundfile as sf


SAMPLING_FREQUENCY = 16000
FFT_LENGTH = 400
FFT_SHIFT = 160
CHANMEL_INDEX = [0, 2, 4, 6]
CHANMEL_INDEX = [0, 1,2,3,4,5,6,7]

def multi_channel_read(prefix=r'C:\Users\a-kojima\Documents\work_python\minatoku_go_bat\sample_data\20G_20GO010I_STR.CH{}.wav',
                       channel_index_vector=np.array([1, 2, 3, 4, 5, 6])):
    wav, _ = sf.read(prefix.replace('{}', str(channel_index_vector[0])), dtype='float32')
    wav_multi = np.zeros((len(wav), len(channel_index_vector)), dtype=np.float32)
    wav_multi[:, 0] = wav
    for i in range(1, len(channel_index_vector)):
        wav_multi[:, i] = sf.read(prefix.replace('{}', str(channel_index_vector[i])), dtype='float32')[0]
    return wav_multi

def apply_range_norm(specs):
    specs = ((specs - np.min(specs)) / (np.max(specs) - np.min(specs))) * (1 - 0) + 0
    return specs


multi_channels_data = sf.read(r'./data/sample_sp.wav')[0][:, CHANMEL_INDEX]
noise = sf.read(r'./data/noise_back.wav')[0][:, CHANMEL_INDEX]

# adjust size
min_size = np.min((np.shape(multi_channels_data)[0], np.shape(noise)[0]))
multi_channels_data = multi_channels_data[0:min_size, :]
noise = noise[0:min_size, :]

noise = noise / np.max(np.abs(noise)) * 0.2
multi_channels_data = multi_channels_data / np.max(np.abs(multi_channels_data)) * 0.7

noise_rand = np.random.normal(loc=0, scale=0.00001, size=(min_size, len(CHANMEL_INDEX)))
synth_r = noise + multi_channels_data + noise_rand

complex_spectrum, _ = util.get_3dim_spectrum_from_data(synth_r, FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)


complex_spectrum_noise, _ = util.get_3dim_spectrum_from_data(noise, FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)
complex_spectrum_speech, _ = util.get_3dim_spectrum_from_data(multi_channels_data, FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)
mask = np.abs(complex_spectrum_speech[0, :, :]) / (np.abs(complex_spectrum_speech[0, :, :]) + np.abs(complex_spectrum_noise[0, :, :]))



number_of_frame = np.shape(complex_spectrum)[1]


beamformer_maker = fbf_mvdr.frame_by_frame_mvdr(SAMPLING_FREQUENCY,
                 FFT_LENGTH,
                 FFT_SHIFT,
                 len(CHANMEL_INDEX))
synth_r = multi_channels_data+noise+noise_rand

# ========================================
# frame by frame 
# ========================================
synth = multi_channels_data[:, 0] * 0
st = 0
ed = FFT_LENGTH
number_of_update = 0
gg = []
beamformer = np.ones((len(CHANMEL_INDEX), FFT_LENGTH // 2 + 1), dtype=np.complex64)
for i in range(0, number_of_frame):
    beamformer_maker.update_param(mask[i, :], np.expand_dims(complex_spectrum[:, i, :], 1))
    number_of_update = number_of_update + 1
    #beamformer = beamformer_maker.get_mvdr_beamformer_by_higuchi()
    beamformer, c = beamformer_maker.get_mvdr_beamformer_by_higuchi_snr_selection()
    #print(beamformer)
    enhanced_speech = beamformer_maker.apply_beamformer(beamformer, complex_spectrum[:, i, :])
    enhanced_speech[np.isnan(enhanced_speech)] = 0
    synth[st:ed] = synth[st:ed] + enhanced_speech 
    st = st + FFT_SHIFT
    ed = ed + FFT_SHIFT

synth = synth / np.max(np.abs(synth)) * 0.8
synth_r_1 = synth_r[:, 0]
synth_r_1 = synth_r_1 / np.max(np.abs(synth_r_1)) * 0.8

sf.write('enhan.wav',synth , SAMPLING_FREQUENCY)
sf.write('noisy.wav', synth_r_1 , SAMPLING_FREQUENCY)







