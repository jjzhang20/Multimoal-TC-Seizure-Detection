#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:07:52 2022

@author: jzhang1
"""

import numpy as np
import sklearn
import csv
import ray
import joblib
import os

import random
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from DataLoading import DataLoading

#  
ray.init()

@ray.remote
class MultiModal():
    def __init__(self):
        
        self.cv_n_folds = 10
        self.SD = 2
        self.k_features = tuple([6, 42])
        self.downsample_rate = 20
        self.C_list = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        self.gamma_list = [ 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        self.DataLoading = DataLoading() 
        self.Load_Samples = {'EEG':DataLoading().Load_EEG_Samples,
                            'EMG':DataLoading().Load_EMG_Samples,
                            'ACC':DataLoading().Load_ACC_Samples,
                            'EEG_ACC':DataLoading().Load_EEG_ACC_Samples,
                            'ECG':DataLoading().Load_ECG_Samples}
                            
        self.results_path = r'/esat/biomeddata/SeizeIT2/Code/Jingwei/EMG/LOSO6/Results'
        self.classifier_path = r'/esat/biomeddata/SeizeIT2/Code/Jingwei/EMG/LOSO6/Classifiers'

    def standard_scale(self, X_train):
        preprocessor = preprocessing.StandardScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)
        return X_train, preprocessor
    
    def optimize_grid(self, clf, TrainingData, TrainingLabels):

        TP = make_scorer(self.tp)
        FN = make_scorer(self.fn)
        TN = make_scorer(self.tn)
        FP = make_scorer(self.fp)        
      
        sensitivity = np.zeros([len(self.C_list), len(self.gamma_list)])
        specificity = np.zeros([len(self.C_list), len(self.gamma_list)])
        for i in range(len(self.C_list)):
            for j in range(len(self.gamma_list)):

                clf = clf.set_params(C = self.C_list[i] , gamma = self.gamma_list[j], class_weight = 'balanced')
                tp, fn, tn, fp = self.cross_validation(clf, TrainingData, TrainingLabels, cv = 5)
               
                sensitivity[i,j] = tp/(tp+fn)
                specificity[i,j] = tn/(tn+fp)
                
        print('sensitivity \n',sensitivity)
        print('specificity \n',specificity)
        sens_index = sensitivity >= 0.8
        if sens_index.any():
            value = np.max(specificity[sens_index])
            index = np.argwhere(specificity == value)
        else:
             value = np.max(sensitivity)
             index = np.argwhere(sensitivity == value)
        
        row = index[0, 0]
        colum = index[0, 1]
        C = self.C_list[row]
        gamma = self.gamma_list[colum]
        
        return C, gamma
    
    def cross_validation(self, clf, Data, Labels, cv = 5):

        Subjects = list(Data.keys())
        random.Random(1).shuffle(Subjects)

        TestSets = self.split_list(Subjects, cv)

        TPs = 0
        FNs = 0
        TNs = 0
        FPs = 0

        for TestSet in TestSets:
            TrainingSet = list(set(Subjects) - set(TestSet))

            Train_X = {subj:data for subj, data in Data.items() if subj in TrainingSet}
            Train_Y = {subj:labels for subj, labels in Labels.items() if subj in TrainingSet}

            Test_X = {subj:data for subj, data in Data.items() if subj in TestSet}
            Test_Y = {subj:labels for subj, labels in Labels.items() if subj in TestSet}

            Train_X, Train_Y = self.DataStack(Train_X, Train_Y)
            Test_X, Test_Y = self.DataStack(Test_X, Test_Y)

            Train_X, Preprocesser = self.standard_scale(Train_X)  
            Test_X = Preprocesser.transform(Test_X)  
            clf.fit(Train_X, Train_Y)
            predict_labels = clf.predict(Test_X)

            tn, fp, fn, tp = confusion_matrix(Test_Y, predict_labels).ravel()

            TPs = TPs + tp
            FNs = FNs + fn
            TNs = TNs + tn
            FPs = FPs + fp
        
        return TPs, FNs, TNs, FPs
            
    def DataStack(self, Data, Labels):

        keys = list(Data.keys())

        for i, key in enumerate(keys):
            if i == 0:
                feat = Data[key]
                label = Labels[key]
            else:
                feat = np.vstack((feat, Data[key]))
                label = np.hstack((label, Labels[key]))
        
        return feat, label
    
    def split_list(self, arr, n):

        return list(np.array_split(arr, n))

    def specificity(self, true_labels, predict_labels):
        
        tn, fp, fn, tp = confusion_matrix(true_labels, predict_labels).ravel()
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        
        return specificity

    def tn(self, true_labels, predict_labels):
        
        tn, fp, fn, tp = confusion_matrix(true_labels, predict_labels).ravel()
        
        return tn
            
    def fp(self, true_labels, predict_labels):
        
        tn, fp, fn, tp = confusion_matrix(true_labels, predict_labels).ravel()
        
        return fp     
    
    def fn(self, true_labels, predict_labels):
        
        tn, fp, fn, tp = confusion_matrix(true_labels, predict_labels).ravel()

        return fn    
    
    def tp(self, true_labels, predict_labels):
        
        tn, fp, fn, tp = confusion_matrix(true_labels, predict_labels).ravel()
        
        return tp   
    
    
    def train_classifier(self, TrainingData, TrainingLabels):
        
        Train_X, Train_Y = self.DataStack(TrainingData, TrainingLabels)
    
        Train_X, Preprocesser = self.standard_scale(Train_X)    
        
        c, gamma = self.optimize_grid(svm.SVC(), TrainingData, TrainingLabels)
        model = svm.SVC(C = c, gamma = gamma, probability = True, class_weight = 'balanced')
        model.fit(Train_X, Train_Y)
        
        return Preprocesser, model

    def save_classifier(self, Subject, Classifier, modality):
        folder = self.classifier_path + '/' + Subject
        if os.path.exists(folder):
            pass
        else:
            os.makedirs(folder)
        
        name = self.classifier_path + '/' + Subject + '/classifier-' + modality + '-' + Subject + '.pkl'
        joblib.dump(Classifier, name)

    def save_preprocesser(self, Subject, Preprocesser, modality):
        folder = self.classifier_path + '/' + Subject
        if os.path.exists(folder):
            pass
        else:
            os.makedirs(folder)

        name = self.classifier_path + '/' + Subject + '/preprocesser-' + modality + '-' + Subject + '.pkl'
        joblib.dump(Preprocesser, name)

    def save_predictpros(self, Subject, rec, predictpros, modality):
        
        if os.path.exists(self.results_path + '/' + Subject):
           pass
        else:
            os.makedirs(self.results_path + '/' + Subject)
        if self.SD == 1:
            name = self.results_path + '/' + Subject + '/' + rec + '_Results_' + modality + '.npy'
        else: 
            name = self.results_path + '/' + Subject + '/' + rec + '_Results_' + modality + '_SD.npy'
        np.save(name, predictpros)
        
    def one_subject_out_cross_validation(self, SubjectsList, TestingSubject, *args):
        
        TrainingSubjects = SubjectsList.copy()
        
        if TestingSubject != 'SUBJ-6-357':
            TrainingSubjects.remove('SUBJ-6-357')

        TrainingSubjects.remove(TestingSubject)
        Train = self.DataLoading.Load_Training_Samples(TrainingSubjects, 
                                                        5, 
                                                        args)   
        print('start to train for subject: ', TestingSubject)
        for modality in args:
            if modality != 'ECG':
                Train_X = Train[modality]
                Train_Y = Train['Labels']
                load_test_samples = self.Load_Samples[modality]
            elif modality == 'ECG':   
                Train_X = Train[modality]
                Train_Y = Train['Labels_ECG']
                load_test_samples = self.Load_Samples[modality]
            
            Preprocesser, Classifier = self.train_classifier(Train_X, Train_Y)
            
            self.save_preprocesser(TestingSubject, Preprocesser, modality)
            self.save_classifier(TestingSubject, Classifier, modality)
            Test_X = load_test_samples(TestingSubject)
            for rec, data in Test_X.items():
                data = Preprocesser.transform(data)
                predictpros = Classifier.predict_proba(data)
                

                self.save_predictpros(TestingSubject, rec, predictpros, modality)


    def get_detections(self, Subject, *args):

        for modality in args:
            classifier_name = self.classifier_path + '/' + Subject + '/classifier-' + modality + '-' + Subject + '.pkl'            
            preprocesser_name = self.classifier_path + '/' + Subject + '/preprocesser-' + modality + '-' + Subject + '.pkl'
            preprocesser = joblib.load(preprocesser_name)
            classifier = joblib.load(classifier_name)
            load_test_samples = self.Load_Samples[modality]
            Test_X = load_test_samples(Subject)
            for rec, data in Test_X.items():
                data = preprocesser.transform(data)
                predictpros = classifier.predict_proba(data)
                self.save_predictpros(Subject, rec, predictpros, modality)

if __name__ == '__main__':
    
    # MultiModal= MultiModal()
    
    SubjectsList = ['SUBJ-1a-025','SUBJ-1a-163','SUBJ-1a-177','SUBJ-1a-224',
                    'SUBJ-1a-226', 'SUBJ-1a-349','SUBJ-1a-353','SUBJ-1a-358',
                    'SUBJ-1a-382', 'SUBJ-1a-414','SUBJ-1a-434','SUBJ-1a-471',
                    'SUBJ-1b-178','SUBJ-1b-307','SUBJ-4-198','SUBJ-4-203',
                    'SUBJ-4-305','SUBJ-4-466','SUBJ-5-365','SUBJ-6-256',
                    'SUBJ-6-275','SUBJ-6-276','SUBJ-6-291','SUBJ-6-357',
                    'SUBJ-6-430','SUBJ-6-463','SUBJ-6-483']

    # multimodals = [MultiModal.remote() for i in range(len(SubjectsList))]
    # predicts = [multi.get_detections.remote(SubjectsList[isubj], 'EEG', 'EMG', 'EEG_ACC', 'ACC', 'ECG') for [isubj, multi] in enumerate(multimodals)]
    # ray.get(predicts)
    # MultiModal.one_subject_out_cross_validation(SubjectsList, SubjectsList[1],'EEG','EMG','EEG_ACC')

    multimodals = [MultiModal.remote() for i in range(len(SubjectsList))]
    Predicts = [multi.one_subject_out_cross_validation.remote(SubjectsList, SubjectsList[isubj],'EEG','EMG','EEG_ACC','ECG') for [isubj, multi] in enumerate(multimodals)]
    ray.get(Predicts)
    