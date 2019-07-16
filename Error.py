# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:17:28 2016

@author: aellenso

"""
import math
import numpy as np


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers),1)

def getStats(meas,model):
    "Takes a measured and modeled series and returns RMSE, bias, PE, SI, and corr_coeff"
    error = meas - model
    RMSE = math.sqrt(mean(error**2))
    bias = np.nanmean(-1*error)
    PE = math.sqrt(np.nanmean((error/meas)**2))*100
    SI = RMSE/np.nanmean(meas)
    corr_coeff = sum((meas - np.nanmean(meas))*(model-np.mean(model)))/math.sqrt(sum((meas-np.mean(meas))**2)*sum((model-np.mean(model))**2))
    return RMSE,bias,PE,SI,corr_coeff
