# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 19:02:58 2021

@author: AD
"""

import numpy as np
from post_processing import post_processing


def evaluation(Predicts, Label_test, ground_truth):

    starts = []
    stops = []
    TP = 0
    FN = 0
    FP = 0
    Delay = []
    FP_start = []
    FP_stop = []
    Detections_start = []
    Detections_stop = []
    max_d = 10
    
    Predictions = np.zeros(Predicts.shape)
    for p in range(len(Predicts)):
        if Predicts[p] == 1 and Predicts[p-1] == 0:
            Predictions[p] = 1

    for i in range(1,len(Label_test)):
        if Label_test[i-1] == 0 and Label_test[i] == 1:
            starts.append(i)
        elif Label_test[i-1] == 1 and (Label_test[i] == 0 or i == len(Label_test)):
            stops.append(i)

    num_seizes = len(starts)
    if num_seizes > 0:
        for iseize in range(num_seizes):
            start = starts[iseize]
            stop = stops[iseize]
            if sum(Predictions[start:stop]) > 0:
                TP = TP + 1
                Delay.append(np.where(np.array(Predictions[start:stop]) == 1)[0][0])
                for k in range(len(Predictions[start:stop])):
                    if Predictions[start + k] == 1 and Predictions[start + k - 1] == 0:
                        Detections_start.append(start + k)
                    if Predictions[start + k] == 0 and Predictions[start + k - 1] == 1:
                        Detections_stop.append(start + k)
                
            else: 
                FN = FN + 1

    if (num_seizes - TP) != FN:
        print('incorrect detection!')
        return 
        

    Predicts[ground_truth == 1] = 0


    for i in range(len(Predicts)):
        if i == 0:
            if Predicts[i] == 1:
                FP = FP + 1
                FP_start.append(i)
        else:
            if (Predicts[i] == 1 and Predicts[i - 1] == 0):
                FP = FP + 1
                FP_start.append(i)
                Detections_start.append(i)
            if i < len(Predicts) - 1:
                if ((Predicts[i] == 1 and Predicts[i + 1] == 0) and 
                    (int(sum(Predicts[FP_start[FP - 1]:i])) == int(i - FP_start[FP - 1]))):
                    FP_stop.append(i)
                    Detections_stop.append(i)
            elif i == len(Predicts) - 1: 
                if ((Predicts[i] == 1) and 
                    (int(sum(Predicts[FP_start[FP - 1]:i])) == int(i - FP_start[FP - 1]))):
                    FP_stop.append(i)
                    Detections_stop.append(i)


    FP_time = [FP_start, FP_stop]
    Detections = [Detections_start, Detections_stop]

    FalseAlarms = FP/len(Predicts)*3600

    
    
    Evaluation = {'TP': TP, 'FN': FN, 'FP':FP, 'Delay': Delay, 'False Positive Times': FP_time, 'Detections':Detections}
                    
    return Evaluation
    

