#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:27:41 2022

@author: jzhang1
"""

from ast import keyword
import numpy as np
import scipy.io
from data_processing import data_processing
import csv
import os




class DataLoading(object):
    
    def __init__(self):
        
        self.data_path = r'/esat/biomeddata/SeizeIT2/Data_clean/Tonic_clonic'
        self.feat_path =  r'/esat/biomeddata/SeizeIT2/Code/Jingwei/EMG/SD'
        self.EEG_feat_path = r'/esat/biomeddata/SeizeIT2/Code/Jingwei/EMG/SD'
        self.label_path =  r'/esat/biomeddata/SeizeIT2/Code/Jingwei/EMG/SD'
        self.ECG_feat_path =  r'/esat/biomeddata/SeizeIT2/Code/Jingwei/EMG/SD'

    def Load_Flat_Line(self, Subject):
        Flat_Line = {}
        root = self.feat_path + '/' + Subject
        flatline_files = self.get_files('FlatLine', root)
        for irec, file in enumerate(flatline_files):
            name = root + '/' + file
            Flat = np.load(name,  allow_pickle = True).item()
            features = self.Filtering_Nan(Flat['FlatLine'])
            ids = file.split('_')
            key = '_'.join(ids[0:-3])
            Flat_Line.update({key:features})
                    
        return Flat_Line

    def MedianDecayingMemory(self, features):

        feats2num = [10,11,12,13,16,17,18,19,20,21,22,23,32,33]
        
       
        for j in range(features.shape[0]):
            if any(features[j,feats2num] == 0) and j != (features.shape[0] - 1):
                pass
            elif j == (features.shape[0] - 1):
                feats = np.array([])
            else:
                feats = features[j:,:]
                break
            
        if feats.shape[0] > 0:           
            z = np.zeros([feats.shape[0], len(feats2num)])
            for i in range(feats.shape[0]):
                if i == 0:
                    z[i,:] = feats[i,feats2num]
                elif i < 118*2:
                    lambda_factor = 0.92
                    z[i,:] = (1 - lambda_factor)*np.median(feats[0:i, feats2num]) + lambda_factor * z[i - 1, :]
                else:
                    lambda_factor = 0.99
                    z[i,:] = (1 - lambda_factor)*np.median(feats[i - 118 * 2, feats2num]) + lambda_factor * z[i - 1, :]

    
            feats[:,feats2num] = feats[:,feats2num]/z     

            NormalizedFeatures = np.vstack((features[:j, :], feats))
        else:
            NormalizedFeatures = features.copy()

        return NormalizedFeatures   

    def Load_EEG_Samples(self, Subject):

        EEG_Feat = {}
        root = self.EEG_feat_path + '/' + Subject
        eeg_files = self.get_files('EEG', root)
        for irec, file in enumerate(eeg_files):
            name = root + '/' + file
            Data = scipy.io.loadmat(name)
            eeg_features = Data['patient']['EEG_features'][0, 0]
            eeg_features_hf = Data['patient']['EEG_features_HF'][0, 0] 
            eeg_features_entropy = Data['patient']['EEG_features_Entropy'][0, 0] 
            features_eeg = np.hstack((eeg_features, eeg_features_hf, eeg_features_entropy))
            features = self.Filtering_Nan(features_eeg)
            # features = self.MedianDecayingMemory(features)
            ids = file.split('_')
            key = '_'.join(ids[0:-3])
            EEG_Feat.update({key:features})

        return EEG_Feat
    
    def Filtering_Nan(self, Data):

        Data[np.isnan(Data)] = 0
        Data[~np.isfinite(Data)] = 10000
            
        return Data

    def get_files(self, modality, root):
        
        contents = os.listdir(root)
        files = sorted(contents.copy())
        if modality == 'EEG':
            data = [file for index, file in enumerate(files) if (modality in file and  '.mat' in file)]
        elif modality == 'ACC':
            data = [file for index, file in enumerate(files) if (modality in file and  '.npy' in file and 'EEG' not in file)]
        else:
            data = [file for index, file in enumerate(files) if (modality in file and ('.npy' in file or '.mat' in file))]

        return data
    
    def Load_EMG_Samples(self, Subject):
        EMG_Feat = {}
        Wintorem = {}
        root = self.feat_path + '/' + Subject
        emg_files = self.get_files('EMG', root)
        for irec, file in enumerate(emg_files):
            name = root + '/' + file
            EMG = np.load(name,  allow_pickle = True).item()
            features = self.Filtering_Nan(EMG['EMG'])
            wintorems = self.Filtering_Nan(EMG['Wintorem'])
            ids = file.split('_')
            key = '_'.join(ids[0:-3])
            EMG_Feat.update({key:features})
            Wintorem.update({key:wintorems})
                    
        return EMG_Feat

    def Load_ACC_Samples(self, Subject):
        ACC_Feat = {}
        root = self.feat_path + '/' + Subject
        acc_files = self.get_files('ACC', root)
        for irec, file in enumerate(acc_files):
            name = root + '/' + file
            ACC = np.load(name,  allow_pickle = True).item()
            features = self.Filtering_Nan(ACC['ACC'])
            ids = file.split('_')
            key = '_'.join(ids[0:-3])
            ACC_Feat.update({key:features})
                    
        return ACC_Feat

    def Load_EEG_ACC_Samples(self, Subject):
        EEG_ACC_Feat = {}
        root = self.feat_path + '/' + Subject
        emg_files = self.get_files('EEG_ACC', root)
        for irec, file in enumerate(emg_files):
            name = root + '/' + file
            EEG_ACC = np.load(name,  allow_pickle = True).item()
            features = self.Filtering_Nan(EEG_ACC['EEG_ACC'])
            ids = file.split('_')
            key = '_'.join(ids[0:-4])
            EEG_ACC_Feat.update({key:features})
                    
        return EEG_ACC_Feat  

    def Load_ECG_Samples(self, Subject):
        
        ECG_Feat = {}
        root  = self.feat_path + '/'+Subject
        ecg_files = self.get_files('ECG', root)
        for irec, file in enumerate(ecg_files):
            name = root + '/' + file 
            Data = scipy.io.loadmat(name)
            features_ecg = Data['ECG_features']['featurematrix'][0, 0]
            features = self.Filtering_Nan(features_ecg)
            ids = file.split('_')
            key = '_'.join(ids[0:-3])
            ECG_Feat.update({key:features})
                
        return ECG_Feat        


    def Load_Labels(self, Subject):
        Labels_Train = {}
        Labels_Testing = {}
        Ground_Truth = {}
        root = self.label_path + '/' + Subject
        label_files = self.get_files('Labels', root)
        for irec, file in enumerate(label_files):
            name = root + '/' + file
            Labels = np.load(name,  allow_pickle = True).item()
            train = Labels['Labels_Train']
            testing = Labels['Labels_Test']
            groundtruth = Labels['Labels_Groundtruth']
            key = file.split('_')[0] + '_' + file.split('_')[1]
            Labels_Train.update({key:train})
            Labels_Testing.update({key:testing})
            Ground_Truth.update({key:groundtruth})

        return Labels_Train, Labels_Testing, Ground_Truth

    def Feature_Flatten(self, Data):
        Flatten = {}
        first = Data['EMG']
        keys = list(first.keys())
        for modality, feat in Data.items():
            if modality != 'Labels' and modality != 'Labels_ECG' and modality != 'GroundTruth':
                feats = [feat[key] for key in keys]
                features = np.vstack(feats)
                Flatten.update({modality: features})
            else:
                feats = [feat[key] for key in keys]
                features = np.hstack(feats)
                Flatten.update({modality: features})

        return Flatten    
                
    def Select_Training_Samples(self, Feat_EMG = None, Feat_ACC = None, Feat_EEG = None, Feat_ECG = None, trainlabels = None,
                                labels = None, groundtruth = None, imbalance_factor = 1):
        
        if Feat_EMG is not None:
            wlison_amplitude = np.reshape(Feat_EMG[:,1],[len(Feat_EMG[:,1])])
            seizures_index = np.where((labels == 1) & (wlison_amplitude != 0))[0]
        else:
            seizures_index = np.where((labels == 1))[0]

        nonseizures_index = np.where((labels == 0) & (groundtruth == 0))[0]
            
        num_seizures = len(seizures_index)
        num_nonseizures = num_seizures*imbalance_factor 
            
        if num_nonseizures <= len(nonseizures_index):
            num_nonseizures = num_nonseizures
        else:
            num_nonseizures = len(nonseizures_index)
                
        np.random.seed(1)
        downsample_nonseizures_index = np.random.choice(nonseizures_index, size = num_nonseizures, replace = False)
            
        index = seizures_index.tolist() + downsample_nonseizures_index.tolist()

        return index

    def Select_Training_ECG(self, Features_ECG, Labels, Training_Samples_Index):
        
        idx = []
        step = 10
        starts = np.array([i*step for i in range(Features_ECG.shape[0])])
        stops = 60 + np.array([i*step for i in range(Features_ECG.shape[0])])
        
        Training_Samples_Index.sort()
        Training_Samples_Index = np.array(Training_Samples_Index)

        Labels_ECG = np.zeros(Features_ECG.shape[0])

        for i in range(Features_ECG.shape[0]):
            temp = Training_Samples_Index[Training_Samples_Index >= starts[i]]
            if any(temp < stops[i]):

                if all(np.diff(Labels[starts[i]:stops[i]])) == 0:
                    idx.append(i)
                    if np.sum(Labels[starts[i]:stops[i]]) > 0:
                        Labels_ECG[i] = 1
                    else:
                        to_remove = np.where(temp < stops[i])[0]
                        Training_Samples_Index = np.delete(Training_Samples_Index,to_remove[0])

        Selected_ECG = Features_ECG[idx,:]
        Selected_Labels = Labels_ECG[idx]

        return Selected_ECG, Selected_Labels
# 
    
    def Load_Training_Samples(self, SubjectsList, imbalance_factor, args):

        Data = {}
        Feat = {}

        for isubj, Subject in enumerate(SubjectsList):
            
            if 'EMG' in args:
                iFeat_EMG = self.Load_EMG_Samples(Subject)
                Data.update({'EMG': iFeat_EMG})           
                iTrainLabels, iLabels, iGroundTruth = self.Load_Labels(Subject)
                Data.update({'Labels': iLabels})
                Data.update({'GroundTruth': iGroundTruth})
            else:
                print('No EMG, could not select training samples')
                return         

            if 'EEG' in args: 
                iFeat_EEG = self.Load_EEG_Samples(Subject)
                Data.update({'EEG': iFeat_EEG})   

            if 'EEG_ACC' in args:
                iFeat_EEG_ACC = self.Load_EEG_ACC_Samples(Subject)
                Data.update({'EEG_ACC': iFeat_EEG_ACC})
                
            if 'ACC' in args:
                iFeat_ACC = self.Load_ACC_Samples(Subject)
                Data.update({'ACC': iFeat_ACC})
            
            if 'ECG' in args:
                iFeat_ECG = self.Load_ECG_Samples(Subject)
                Data.update({'ECG': iFeat_ECG})

                
            Features = self.Feature_Flatten(Data)

            Training_Samples_Index = self.Select_Training_Samples(Feat_EMG = Features['EMG'], 
                                                                labels = Features['Labels'], 
                                                                groundtruth = Features['GroundTruth'], 
                                                                imbalance_factor = imbalance_factor)
            
            # if 'EMG' in args:
            #     if isubj == 0:
            #         Feat_EMG = Features['EMG'][Training_Samples_Index,:]
            #         Labels = Features['Labels'][Training_Samples_Index]
            #     else:
            #         Feat_EMG = np.vstack((Feat_EMG, Features['EMG'][Training_Samples_Index,:]))
            #         Labels = np.hstack((Labels,Features['Labels'][Training_Samples_Index]))
            
            # if 'EEG' in args:
            #     if isubj == 0:
            #         Feat_EEG = Features['EEG'][Training_Samples_Index,:]
            #     else:
            #         Feat_EEG = np.vstack((Feat_EEG, Features['EEG'][Training_Samples_Index,:]))

            # if 'EEG_ACC' in args:
            #     if isubj == 0:
            #         Feat_EEG_ACC = Features['EEG_ACC'][Training_Samples_Index,:] 
            #     else:
            #         Feat_EEG_ACC = np.vstack((Feat_EEG_ACC, Features['EEG_ACC'][Training_Samples_Index,:]  ))
            
            # if 'ACC' in args:
            #     if isubj == 0:
            #         Feat_ACC = Features['ACC'][Training_Samples_Index,:]
            #     else:
            #         Feat_ACC = np.vstack((Feat_ACC, Features['ACC'][Training_Samples_Index,:]))

            # if 'ECG' in args:
            #     if isubj == 0:
            #         iTraining_Feat_ECG, iLabels_ECG = self.Select_Training_ECG(Features['ECG'], Features['Labels'], Training_Samples_Index)
            #         Feat_ECG = iTraining_Feat_ECG
            #         Labels_ECG = iLabels_ECG
            #     else:
            #         Feat_ECG = np.vstack((Feat_ECG, iTraining_Feat_ECG))
            #         Labels_ECG = np.hstack((Labels_ECG, iLabels_ECG))   

            if 'EMG' in args:
                if isubj == 0:
                    Feat_EMG = {Subject: Features['EMG'][Training_Samples_Index,:]}
                    Labels = {Subject: Features['Labels'][Training_Samples_Index]}
                else:
                    Feat_EMG.update({Subject: Features['EMG'][Training_Samples_Index,:]})
                    Labels.update({Subject: Features['Labels'][Training_Samples_Index]})
            
            if 'EEG' in args:
                if isubj == 0:
                    Feat_EEG = {Subject: Features['EEG'][Training_Samples_Index,:]}
                else:
                    Feat_EEG.update({Subject: Features['EEG'][Training_Samples_Index,:]})

            if 'EEG_ACC' in args:
                if isubj == 0:
                    Feat_EEG_ACC = {Subject: Features['EEG_ACC'][Training_Samples_Index,:]} 
                else:
                    Feat_EEG_ACC.update({Subject: Features['EEG_ACC'][Training_Samples_Index,:]})
            
            if 'ACC' in args:
                if isubj == 0:
                    Feat_ACC = {Subject: Features['ACC'][Training_Samples_Index,:]}
                else:
                    Feat_ACC.update({Subject: Features['ACC'][Training_Samples_Index,:]})

            if 'ECG' in args:
                if isubj == 0:
                    iTraining_Feat_ECG, iLabels_ECG = self.Select_Training_ECG(Features['ECG'], Features['Labels'], Training_Samples_Index)
                    Feat_ECG = {Subject:iTraining_Feat_ECG}
                    Labels_ECG = {Subject: iLabels_ECG}
                else:
                    Feat_ECG.update({Subject: iTraining_Feat_ECG})
                    Labels_ECG.update({Subject: iLabels_ECG})          
                    
        if 'EMG' in args:
            Feat.update({'EMG': Feat_EMG})
            Feat.update({'Labels': Labels})
        if 'EEG' in args:         
            Feat.update({'EEG': Feat_EEG})
   
        if 'EEG_ACC' in args:
            Feat.update({'EEG_ACC': Feat_EEG_ACC})

        if 'ACC' in args:
            Feat.update({'ACC': Feat_ACC})

        if 'ECG' in args:
            Feat.update({'ECG': Feat_ECG})
            Feat.update({'Labels_ECG': Labels_ECG})

        return Feat
       


    

    
