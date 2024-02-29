# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:16:35 2022

@author: AD
"""

import numpy as np
import pandas
import mne
import dit
import EntropyHub
import scipy.io
import scipy.stats
import os
import csv
from data_processing import data_processing
import warnings
import ray

# ray.init(num_cpus = 10)
# # 
# # 
# @ray.remote
class feature_extraction(object):
    def __init__(self):
        self.SD = 2
        self.window_length = 2
        self.overlap = 1
        self.number_acc_features = 13
        self.number_emg_features = 7
        self.data_processing = data_processing()
        self.data_path = r'/esat/biomeddata/SeizeIT2/Data_clean/Tonic_clonic'
        self.label_path =  r'/esat/biomeddata/SeizeIT2/Data_clean/Tonic_clonic'
        self.feature_path = r'/esat/biomeddata/SeizeIT2/Code/Jingwei/EMG/SD'

    def find_channel(self, SD, channels):
        
        channel_ear = []
        channel_emg = []
        channel_acc = []
        channel_eeg_acc = []
        
        if SD == 1:
            emg = ['EMG']
            acc = ['ACC X','ACC Y','ACC Z']
            ear = ['LiOorTop', 'LiOorAcht', 'ReOorTop', 'ReOorAcht']
            for i in range(len(ear)):
                for j in range(len(channels)):
                    if ear[i] in channels[j]:
                        channel_ear.append(j)
            for i in range(len(emg)):
                for j in range(len(channels)):
                    if emg[i] in channels[j] and 'SD' not in channels[j]:
                        channel_emg.append(j)
            for i in range(len(acc)):
                for j in range(len(channels)):
                    if acc[i] in channels[j]:
                        channel_acc.append(j)
                    
        else:
            emg = ['EMG SD']
            acc = ['ECGEMG SD ACC X','ECGEMG SD ACC Y','ECGEMG SD ACC Z']
            acc2 = ['EEG SD ACC X','EEG SD ACC Y','EEG SD ACC Z']
            ear = ['left','right','CROSStop']
            
            for i in range(len(ear)):
                for j in range(len(channels)):
                    if ear[i] in channels[j]:
                        channel_ear.append(j)
                if i == 1 & len(channel_ear) == 2:
                    raise Exception('generalized setup')
                    
            if len(channel_ear) == 0:
                ear = ['Ch2 EEG','Ch1 EEG']
                for i in range(len(ear)):
                    for j in range(len(channels)):
                        if channels[j] in ear[i]:
                            channel_ear.append(j)
                            
            for i in range(len(emg)):
                for j in range(len(channels)):
                    if emg[i] in channels[j] and 'ECG' not in channels[j]:
                        channel_emg.append(j)
                        
            for i in range(len(acc)):
                for j in range(len(channels)):
                    if acc[i] in channels[j]:
                        channel_acc.append(j) 
                    elif acc2[i] in channels[j]:
                        channel_eeg_acc.append(j)
        
        return channel_ear, channel_emg, channel_acc, channel_eeg_acc

    def read_edf(self, root):
        SD = self.SD
        
        raw_data = mne.io.read_raw_edf(root)
        channels = raw_data.ch_names
        fs = raw_data.info['sfreq']
        channel_ear, channel_emg, channel_acc, channel_eeg_acc = self.find_channel(SD, channels)
        
        try:
            eeg, _ = raw_data[channel_ear,:]
            eeg = 10**6*eeg
        except:
            if len(channel_ear) == 0:
                warnings.warn('No EEG channels!')
                eeg = []   
        try:
            emg, _ = raw_data[channel_emg,:]
            emg = 10**6*emg
        except:
            if len(channel_emg) == 0:
                warnings.warn('No EMG channels!')
                emg = [] 
        try:
            acc, _ = raw_data[channel_acc,:]          
        except:
            if len(channel_acc) == 0:
                warnings.warn('No ACC channels!')
                acc = [] 
        
        try:
            eeg_acc, _ = raw_data[channel_eeg_acc,:]          
        except:
            if len(channel_eeg_acc) == 0:
                warnings.warn('No EEG_ACC channels!')
                eeg_acc = [] 
       
        return eeg, emg, acc, eeg_acc, fs
    
    def creat_labels(self, features, subject, annot_groundtruth, annot_test, annot_train):
        annots_groundtruth = self.label_path + '/' + subject + '/' + annot_groundtruth
        annots_test = self.label_path + '/' + subject + '/' + annot_test
        annots_train = self.label_path + '/' + subject + '/' + annot_train
        
        labels_groundtruth = np.zeros(features.shape[0])
        labels_test = np.zeros(features.shape[0])
        labels_train = np.zeros(features.shape[0])
         
        with open(annots_groundtruth) as annotations:
            annot_reader = csv.reader(annotations, delimiter = '\t')
            for index, row in enumerate(annot_reader):
                if index > 4:
                    if row[2] == 'seizure': 
                        onset = int(row[0])
                        stop = int(row[1])
                        if onset > 0:
                            labels_groundtruth[onset - 1 : stop - 1] = 1       
                            
        with open(annots_test) as annotations:
            annot_reader = csv.reader(annotations, delimiter = '\t')
            for index, row in enumerate(annot_reader):
                if index > 4:
                    if row[2] == 'Tonic-clonic movements': 
                        onset = int(row[0])
                        stop = int(row[1])
                        if onset > 0:
                            labels_test[onset - 1 : stop - 1] = 1 
         
        with open(annots_train) as annotations:
            annot_reader = csv.reader(annotations, delimiter = '\t')
            for index, row in enumerate(annot_reader):
                if index > 4:
                    if row[2] == 'Tonic-clonic movements': 
                        onset = int(row[0])
                        stop = int(row[1])
                        if onset > 0:
                            labels_train[onset - 1 : stop - 1] = 1                            
             
        return labels_groundtruth, labels_test, labels_train
    
    def emg_zerocrossing(self, data):
        
        data = data - np.mean(data)
        difference = np.diff(np.sign(data) > 0)
        
        EMGzc = np.sum(abs(difference))
        
        return EMGzc
    
    def emg_wilsonamplitude(self, data):
        
        difference = np.diff(data)
        EMGwa = len([j for j in difference if j > 50])
        
        return EMGwa
    
    def emg_wintorem(self, data):
    
       EMGwintorem =  np.max(np.std(data.T))
        
       return EMGwintorem
    
    def emg_mean_abs(self, data):
        
        EMGmean = np.mean(abs(data))
        
        return EMGmean
    
    def emg_std(self, data):
        
        EMGstd = np.std(data)
        
        return EMGstd
    
    def emg_median(self, data):
        
        EMGmedian = np.median(abs(data))
        
        return EMGmedian
    
    def emg_wave_length(self, data):
        
        delta = np.diff(data)
        EMGwl = np.sum(delta)
        
        return EMGwl
    
    def emg_mean_frequency(self, data, fs):
        
        bins = len(data)
        
        fft = np.fft.fft(data, n = bins)[1:bins//2]        
        freq = np.fft.fftfreq(bins, d = 1/fs)[1:bins//2]
        energy = np.square( np.abs(fft))/bins
        
        EMGmean_freq = np.sum(energy * freq) / np.sum(energy)
        
        return EMGmean_freq
    
    def emg_frequency_percentage(self, data, fs):

        bins = len(data)
        fft = np.fft.fft(data, n = bins)[1:bins//2]        
        energy = np.square( np.abs(fft))/bins
        
        segment = len(energy)//9
        freq_energy = np.sum(np.reshape( energy[:segment * 9], [9,segment]), axis = 1)
        energy_percentage = freq_energy/np.sum(energy[:segment * 9])
        
        return energy_percentage
    
    def emg_CREn(self, data):
        
        data[data < 10 ** -5] = 0
        if np.sum(abs(np.diff(data))) <= 0.01:
            CREn = 0
        else:
            number_bins = 20
            step = (np.max(data) - np.min(data))/(number_bins)
            bins = np.arange(np.min(data), np.max(data) + step, step)
            bin_average = [np.mean([bins[i],bins[i+1]]) for i in range(len(bins) - 1)]
            bins[-1] = bins[-1] + 0.0001
            segments = pandas.cut(data, bins, right = False)
            counts = pandas.value_counts(segments, sort = False)/len(data)
            px = [i for i in counts]
            distribution = dit.ScalarDistribution(bin_average, px)
            CREn = dit.other.cumulative_residual_entropy(distribution)
        
        return CREn
    
    def emg_FuzzEn(self, data):
        
        EMG_FuzzEn,_ ,_ = EntropyHub.FuzzEn(data, m = 2, tau = 2, r = (0.3, 2))
                                                      
        return EMG_FuzzEn
    
    def acc_iqr(self, data):
        
        ACCiqr = scipy.stats.iqr(data)

        return ACCiqr
        
    def acc_std(self, data):
        
        ACCstd = np.std(data)
        
        return ACCstd
    
    def acc_max(self, data):
        
        ACCmax = np.max(abs(data))
        
        return ACCmax
    
    def acc_mean(self, data):
        
        ACCmean = np.mean(data)
        
        return ACCmean
        
    def acc_median(self, data):
        
        ACCmedian = np.median(abs(data))
        
        return ACCmedian
    
    def acc_zerocrossing(self, data):
        
        data = data - np.mean(data)
        difference = np.diff(np.sign(data) > 0)
        
        ACCzc = np.sum(abs(difference))
        
        return ACCzc
    
    def acc_kurtosis(self, data):
        
        ACCkurtosis = scipy.stats.kurtosis(data)
        
        return ACCkurtosis
    
    def acc_skewness(self, data):
        
        ACCskewness = scipy.stats.skew(data)
        
        return ACCskewness
    
    def acc_averpower(self, data):
        
        ACCaverpower = np.mean(data ** 2)
        
        return ACCaverpower
    
    def acc_poincare(self, data):
        
        # result = pyhrv.nonlinear.poincare(data)
        	
        x1 = np.asarray(data[:-1])
        x2 = np.asarray(data[1:])
        x3 = np.vstack((x1,x2,np.ones(len(x1))))
    
    	# SD1 & SD2 Computation
        sd1 = np.std(np.subtract(x1, x2) / np.sqrt(2))
        sd2 = np.std(np.add(x1, x2) / np.sqrt(2))
        s = np.pi*sd1*sd2
        a = [np.linalg.det(x3[:,i:i+3]) for i in range(len(x1) - 2)]
        ccm = np.sum(a)/(s*(len(x1) - 1))    
        
        ACCsd1 = sd1
        ACCsd2 = sd2
        ACCsdratio = sd1/sd2
        ACCccm = ccm


        return ACCsd1, ACCsd2, ACCsdratio, ACCccm

    def eeg_flatline(self, data):
        
        difference_1 = np.diff(data[:-3])
        difference_1[abs(difference_1) < 5] = 0
        difference_2 = np.diff(data[1:-2])
        difference_2[abs(difference_2) < 5] = 0 
        difference_3 = np.diff(data[2:-1])
        difference_3[abs(difference_3) < 5] = 0 
        difference_4 = np.diff(data[3:])
        difference_4[abs(difference_4) < 5] = 0 

        flats = (difference_1 == 0) & (difference_2 == 0) & (difference_3 == 0) & (difference_4 == 0)
        num_flats = sum(flats)
        ratio_flat= num_flats/len(data)
        
        return ratio_flat
                                    
    def feature_calculation_emg(self, data, fs):
        print('datalength:',data.shape)
        data = self.data_processing.preprocessing_emg(data, [20,10,0.1,40], 50, 0.95, fs)
        data_length = int(data.shape[1]/fs)
        step = self.window_length - self.overlap
        windows = np.array([range(0, data_length + 1 - self.window_length, step),
                            range(self.window_length, data_length + 1, step)])

        for ch in range(data.shape[0]):
            print('EMG Channel:', ch + 1)
            
            # zero_crossing = np.zeros([windows.shape[1] ,1])
            # wilson_amplitude = np.zeros([windows.shape[1] ,1])
            ch_features = np.zeros([windows.shape[1], self.number_emg_features])
            ch_wintorem = np.zeros([windows.shape[1], 1] )
            
            for i in range(windows.shape[1]):
                data_segment = data[ch, windows[0,i]*int(fs):windows[1,i]*int(fs)]

                zero_crossing = self.emg_zerocrossing(data_segment)
                wilson_amplitude = self.emg_wilsonamplitude(data_segment)
                mean_abs = self.emg_mean_abs(data_segment)
                variance = self.emg_std(data_segment)
                median_abs = self.emg_median(data_segment)
                FEn = self.emg_FuzzEn(data_segment)
                CREn = self.emg_CREn(data_segment)

                ch_features[i,:] = np.hstack((zero_crossing, wilson_amplitude, mean_abs, variance, 
                                              median_abs,FEn[1], CREn))
                
                ch_wintorem[i,:] = self.emg_wintorem(data_segment)

            # ch_features = np.hstack((zero_crossing, wilson_amplitude))             
            if ch == 0:
                features = ch_features
                wintorem = ch_wintorem
            else:
                features = np.concatenate((features, ch_features), axis = 1)
                wintorem = np.concatenate((wintorem, ch_wintorem), axis = 1)
                
        return features, wintorem
    
    def feature_calculation_acc(self, data, fs):
        
        data = self.data_processing.preprocessing_acc(data, [2, 25], fs)
        
        data_norm = np.sqrt(data[0,:] ** 2 + data[1,:] ** 2 + data[2,:] ** 2)         
        data = np.vstack((data, data_norm))

        data_length = int(data.shape[1]/fs)
        step = self.window_length - self.overlap
        
        windows = np.array([range(0, data_length + 1 - self.window_length, step),
                            range(self.window_length, data_length + 1, step)])

        for ch in range(data.shape[0]):
            print('ACC Channel:', ch + 1)  
            ch_features = np.zeros([windows.shape[1], self.number_acc_features])
            for i in range(windows.shape[1]):
                data_segment = data[ch, windows[0,i]*int(fs):windows[1,i]*int(fs)]
                ACCiqr = self.acc_iqr(data_segment)
                ACCstd = self.acc_std(data_segment)
                ACCmax = self.acc_max(data_segment)
                ACCmean = self.acc_mean(data_segment)
                ACCmedian = self.acc_median(data_segment)
                ACCzc = self.acc_zerocrossing(data_segment)
                ACCkurtosis = self.acc_kurtosis(data_segment)
                ACCskewness = self.acc_skewness(data_segment)
                ACCaverpower = self.acc_averpower(data_segment)
                ACCsd1, ACCsd2, ACCsdratio, ACCccm = self.acc_poincare(data_segment)

                ch_features[i,:] = np.array([ACCiqr, ACCstd, ACCmax, ACCmean, ACCmedian, ACCzc,
                                             ACCkurtosis, ACCskewness, ACCaverpower, ACCsd1, 
                                             ACCsd2, ACCsdratio ,ACCccm])    
        
            if ch == 0:
                features = ch_features

            else:
                features = np.concatenate((features, ch_features), axis = 1)
        
        return features

    def feature_calculation_flatline(self, data, fs):
        print('datalength:',data.shape)
        data_length = int(data.shape[1]/fs)
        step = self.window_length - self.overlap
        windows = np.array([range(0, data_length + 1 - self.window_length, step),
                            range(self.window_length, data_length + 1, step)])

        for ch in range(data.shape[0]):
            print('EEG Channel:', ch + 1)
            ch_features = np.zeros([windows.shape[1], 1])
            
            for i in range(windows.shape[1]):
                data_segment = data[ch, windows[0,i]*int(fs):windows[1,i]*int(fs)]

                flatlines = self.eeg_flatline(data_segment)

                ch_features[i,:] = flatlines        
       
            if ch == 0:
                features = ch_features
            else:
                features = np.concatenate((features, ch_features), axis = 1)
                
        return features        
    
    def feature_extractions(self, Subject, *args):
        folder = self.data_path + '/' + Subject
        edfs = self.data_processing.get_edf(folder)
        
        for rec, edf in enumerate(edfs):
            root = folder + '/' + edf
            print('no.',rec + 1,'recording')
            print(root[:-4])
            eeg, emg, acc, eeg_acc, fs = self.read_edf(root)
            annot_groundtruth, annot_test, annot_train = self.data_processing.get_tsv(edf)
            
            if ('EMG' in args) and  (len(emg) != 0):
                rfeatures, rwintorem = self.feature_calculation_emg(emg, fs)
                 
                EMG_Feat = rfeatures
                Wintorem = rwintorem

                EMG= {'EMG': EMG_Feat, 'Wintorem': Wintorem}
                self.save_features(Subject, edf, EMG, 'EMG')
                # name = self.feature_path + '/' + Subject + '/' + edf[:-4] + '_Feat_' + 'EMG' + '_SD.npy'       
                # EMG = np.load(name, allow_pickle = True).item()
                # rfeatures = EMG['EMG']

            if ('ACC' in args) and  (len(acc) != 0):
                rfeatures_acc = self.feature_calculation_acc(acc, fs)  
                Feat_ACC = rfeatures_acc
                
                ACC = {'ACC': Feat_ACC}
                self.save_features(Subject, edf, ACC, 'ACC')

               
            if ('EEG_ACC' in args) and  (len(eeg_acc) != 0):
                rfeatures_eeg_acc = self.feature_calculation_acc(eeg_acc, fs) 
                Feat_EEG_ACC = rfeatures_eeg_acc                
                EEG_ACC = {'EEG_ACC': Feat_EEG_ACC}
                self.save_features(Subject, edf, EEG_ACC, 'EEG_ACC')

            if ('Labels' in args):
                rlabels_groundtruth, rlabels_test, rlabels_train = self.creat_labels(rfeatures, Subject, annot_groundtruth, annot_test, annot_train )
  
                Labels_Groundtruth = rlabels_groundtruth
                Labels_Test = rlabels_test
                Labels_Train = rlabels_train

                Labels = {'Labels_Groundtruth': Labels_Groundtruth, 'Labels_Test': Labels_Test,'Labels_Train': Labels_Train}
                self.save_features(Subject, edf, Labels, 'Labels')

            if ('FlatLine' in args) and  (len(eeg) != 0):
                rfeatures_flatline = self.feature_calculation_flatline(eeg, fs)

                FlatLine= {'FlatLine': rfeatures_flatline}
                self.save_features(Subject, edf, FlatLine, 'FlatLine')

    
    def save_features(self, Subject, edf, subjdata, modality = 'EMG'): 
        
        SD = self.SD
        
        if SD == 1:
            if os.path.isdir(self.feature_path + '/' + Subject):
                pass
            else:
                os.mkdir(self.feature_path + '/' + Subject)
            name = self.feature_path + '/' + Subject + '/' + edf[:-4] + '_Feat_' + modality + '.npy'
        elif SD == 2:
            if os.path.isdir(self.feature_path + '/' + Subject):
                pass
            else:
                os.mkdir(self.feature_path + '/' + Subject)
            name = self.feature_path + '/' + Subject + '/' + edf[:-4] + '_Feat_' + modality + '_SD.npy'       
    
        np.save(name, subjdata)

     
        
if __name__ == '__main__':
# # 
#     SubjectsList = ['SUBJ-1a-025','SUBJ-1a-163','SUBJ-1a-177','SUBJ-1a-224',
#                     'SUBJ-1a-226', 'SUBJ-1a-297','SUBJ-1a-349','SUBJ-1a-353',
#                     'SUBJ-1a-358','SUBJ-1a-382', 'SUBJ-1a-414','SUBJ-1a-434',
#                     'SUBJ-1a-471','SUBJ-1b-178','SUBJ-1b-307','SUBJ-4-198',
#                     'SUBJ-4-203','SUBJ-4-305','SUBJ-4-466','SUBJ-5-365',
#                     'SUBJ-6-256','SUBJ-6-275','SUBJ-6-276','SUBJ-6-291',
#                     'SUBJ-6-357','SUBJ-6-430','SUBJ-6-463','SUBJ-6-483']   

    SubjectsList = ['SUBJ-1a-224']   

    feature_extraction().feature_extractions(SubjectsList[0],'EMG','EEG_ACC','ACC','Labels')
    # feature_nodes = [feature_extraction.remote() for i in range(len(SubjectsList))]
    # features = [feat.feature_extractions.remote(SubjectsList[i],'FlatLine') for i, feat in enumerate(feature_nodes)]
    # ray.get(features)     

        