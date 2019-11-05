# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:40:29 2019

@author: a-kojima
"""
import numpy as np
import copy
from . import util

"""
FRAME-BY-FRAME CLOSED-FORM UPDATE FOR MASK-BASED
ADAPTIVE MVDR BEAMFORMING

"""

class frame_by_frame_mvdr:
    
    def __init__(self,
                 sampling_frequency,
                 fft_length,
                 fft_shift,
                 number_of_channel,
                 reference_mic=0,
                 condition_number_inv_threshold=10**(-6), # -6
                 scm_inv_threshold=10**(-2), # -10
                 beamformer_inv_threshold=10**(-2)):         # -6
        self.sampling_frequency=sampling_frequency
        self.fft_length=fft_length
        self.fft_shift=fft_shift
        self.condition_number_inv_threshold=condition_number_inv_threshold
        self.scm_inv_threshold=scm_inv_threshold
        self.beamformer_inv_threshold=beamformer_inv_threshold
        self.number_of_channel = number_of_channel
        self.reference_mic = reference_mic
        
        # SCM init
        self.number_of_bins = self.fft_length // 2 + 1
        
        self.R_s = np.zeros((self.number_of_channel, self.number_of_channel, self.number_of_bins), dtype=np.complex64)
        self.R_n = np.zeros((self.number_of_channel, self.number_of_channel, self.number_of_bins), dtype=np.complex64)
        #self.Y_inv = np.zeros((self.number_of_channel, self.number_of_channel, self.number_of_bins), dtype=np.complex64)
        self.Y_inv = np.repeat(np.expand_dims(np.eye(self.number_of_channel, dtype=np.complex64), 2), self.number_of_bins, axis=2)
        self.Y = np.zeros((self.number_of_channel, self.number_of_channel, self.number_of_bins), dtype=np.complex64)
        self.beamformer = np.zeros((self.number_of_channel, self.number_of_bins), dtype=np.complex64)        
    
    def init_Y_inv(self, speech_mask, complex_spectrum):
        """
        this method should be called at first frame
        """
        for f in range(0, self.number_of_bins):   
            h = np.multiply.outer(complex_spectrum[:, 0, f], np.conj(complex_spectrum[:, 0, f]).T)            
            self.Y_inv[:, :, f] = np.linalg.pinv(h, rcond=self.scm_inv_threshold)
        
        
        
    def update_param(self, speech_mask, complex_spectrum):        
        """
        speech_mask: (FFTL // 2 + 1, 1)
        complex_specetrum_frame: (N of CHANNEL, 1, FFTL // 2 + 1) 

        """   
        # get SCM at current frane
        R_s, Y = self.get_scm(complex_spectrum, speech_mask)
        
        # update Y (6)
        old_Y = copy.deepcopy(self.Y)
        self.Y = self.Y + Y
                        
        # update Y-1 (7)
        for f in range(0, self.number_of_bins):
            Y_inv = np.linalg.pinv(old_Y[:, :, f], rcond=self.scm_inv_threshold)
            w1 = np.matmul(Y_inv, complex_spectrum[:, :, f])
            w1 = np.matmul(w1, np.conj(complex_spectrum[:, :, f]).T)
            w1 = np.matmul(w1, Y_inv)
            w2 = np.matmul(np.conj(complex_spectrum[:, :, f]).T, Y_inv)
            w2 = np.matmul(w2, complex_spectrum[:, :, f])
            w2 = np.complex(1) + w2
            weight = w1 / w2 # (CH * CH)
            self.Y_inv[:, :, f] = Y_inv - weight
        
        # update R (8)
        self.R_s = self.R_s + R_s
        
        # in order to calculate SNR, save noise SCM
        #R_n, _ = self.get_scm(complex_spectrum, 1 - speech_mask)
        #self.R_n = self.R_n + R_n
        
    def get_scm(self, complex_spectrum, speech_mask):
        """
        return SCM at current frame
        """
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)                
        Y = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)
        R = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)        
        # init R_noisy and R_noise
        for f in range(0, number_of_bins):            
            h = np.multiply.outer(complex_spectrum[:, 0, f], np.conj(complex_spectrum[:, 0, f]).T)
            R[:, :, f] = h * speech_mask[f]
            Y[:, :, f] = h 
        return R, Y
                    
    def get_mvdr_beamformer_by_higuchi(self, reference_channel_index=0):                
        """ this beamformer use fixed mic
        """
        d = np.zeros(self.number_of_channel, dtype=np.complex64)
        d[reference_channel_index] = 1 # sample       
        #d = np.ones(self.number_of_channel, dtype=np.complex64)
        # (9)
        for f in range(0, self.number_of_bins):
            w1 = np.matmul(self.Y_inv[:, :, f], self.R_s[:, :, f])
            w1_2 = np.matmul(w1, d)
            w2 = np.trace(w1)
            w2 = np.reshape(w2, [1, 1])
            #print('1', w1_2)
            #print('2', w2)
            #print('yinv', self.Y_inv)
            #print('rs', self.R_s)
            w = w1_2 / w2
            w = np.reshape(w, self.number_of_channel)
            self.beamformer[:, f] = w
        return self.beamformer    
    
    def _get_mvdr_beamformer_by_higuchi_snr_selection(self, max_snr_channel=0):     
        beamformer_tmp = np.ones((self.number_of_channel, self.number_of_channel, self.number_of_bins), dtype=np.complex64)             
        selected_SNR = np.zeros(self.number_of_channel, dtype=np.float32)
        
        for c in range(0, self.number_of_channel):
            d = np.zeros(self.number_of_channel, dtype=np.complex64)
            d[c] = 1 # sample        
            # (9)
            for f in range(0, self.number_of_bins):
                w1 = np.matmul(self.Y_inv[:, :, f], self.R_s[:, :, f])
                w1_2 = np.matmul(w1, d)
                w2 = np.trace(w1)
                w2 = np.reshape(w2, [1, 1])
                w = w1_2 / w2
                w = np.reshape(w, self.number_of_channel)
                beamformer_tmp[c, :, f] = w
            w1_sum = 0
            w2_sum = 0
            for f2 in range(0, self.number_of_channel):
                snr_post_w1 = np.matmul(np.conjugate(beamformer_tmp[c, :, f2]).T, self.R_s[:, :, f2])
                snr_post_w1 = np.matmul(snr_post_w1, beamformer_tmp[c, :, f2])
                snr_post_w2 = np.matmul(np.conjugate(beamformer_tmp[c, :, f2]).T, self.R_n[:, :, f2])
                snr_post_w2 = np.matmul(snr_post_w2, beamformer_tmp[c, :, f2])
                w1_sum = w1_sum + snr_post_w1
                w2_sum = w2_sum + snr_post_w2
            
            selected_SNR[c] = w1_sum / w2_sum

        #  select beamformer
        #print(selected_SNR)
        max_snr_index = np.argmax(selected_SNR)
        return beamformer_tmp[max_snr_index, :, :], c

    def get_mvdr_beamformer_by_higuchi_snr_selection(self, max_snr_channel=0):     
        beamformer_tmp = np.ones((self.number_of_channel, self.number_of_channel, self.number_of_bins), dtype=np.complex64)             
        selected_SNR = np.zeros(self.number_of_channel, dtype=np.float32)
        
        # in advance, calculate inverse matrix etc...
#        print('inverse')
        w1 = np.zeros((self.number_of_channel, self.number_of_channel, self.number_of_bins), dtype=np.complex64)
        for ii in range(0, self.number_of_bins):
            w1[:, :, ii] = np.matmul(self.Y_inv[:, :, ii], self.R_s[:, :, ii])
        
        for c in range(0, self.number_of_channel):
            d = np.zeros(self.number_of_channel, dtype=np.complex64)
            d[c] = 1 
            # (9)
            for f in range(0, self.number_of_bins):
                #w1 = np.matmul(self.Y_inv[:, :, f], self.R_s[:, :, f])
                w1_tmp = w1[:,:,f]
                w1_2 = np.matmul(w1_tmp, d)
                w2 = np.trace(w1_tmp)
                w2 = np.reshape(w2, [1, 1])
                w = w1_2 / w2
                w = np.reshape(w, self.number_of_channel)
                beamformer_tmp[c, :, f] = w
#            print('design bf done.')
            w1_sum = 0
            w2_sum = 0
            for f2 in range(0, self.number_of_channel):
                snr_post_w1 = np.matmul(np.conjugate(beamformer_tmp[c, :, f2]).T, self.R_s[:, :, f2])
                snr_post_w1 = np.matmul(snr_post_w1, beamformer_tmp[c, :, f2])
                snr_post_w2 = np.matmul(np.conjugate(beamformer_tmp[c, :, f2]).T, self.R_n[:, :, f2])
                snr_post_w2 = np.matmul(snr_post_w2, beamformer_tmp[c, :, f2])
                w1_sum = w1_sum + snr_post_w1
                w2_sum = w2_sum + snr_post_w2
#            print('snr done.')
            selected_SNR[c] = w1_sum / w2_sum

        #  select beamformer
        #print(selected_SNR)
        max_snr_index = np.argmax(selected_SNR)
        return beamformer_tmp[max_snr_index, :, :], c


    def apply_beamformer(self, beamformer, complex_spectrum):
        #number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)  
        complex_spectrum = np.expand_dims(complex_spectrum, 1)
        enhanced_spectrum = np.zeros((1, self.number_of_bins), dtype=np.complex64)
        for f in range(0, self.number_of_bins):
            enhanced_spectrum[:, f] = np.matmul(np.conjugate(beamformer[:, f]).T, complex_spectrum[:, :, f])
        return util.spec2wav(enhanced_spectrum, self.sampling_frequency, self.fft_length, self.fft_length, self.fft_shift)
