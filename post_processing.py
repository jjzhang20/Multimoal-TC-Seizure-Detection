#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:46:39 2022

@author: jzhang1
"""

import numpy as np

def post_processing(Predictions):
    
    Predicted = Predictions[:].copy()

    Detected_temp = np.zeros(Predicted.shape) 

    for p in range(20, len(Predicted)):
        if np.sum(Predicted[p-20:p])>=18:
            Detected_temp[p] = 1

    return Detected_temp