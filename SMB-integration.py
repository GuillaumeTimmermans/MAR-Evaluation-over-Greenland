#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:52:34 2025

@author: gtimmermans
"""

import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset

def smb_computation(file):
    data_nc = Dataset(file)
    try :
        smb, msk, lon, lat, area = [np.array(data_nc[i]) for i in ["MBTO", "MSK","LON","LAT", "AREA"]]
    except( KeyError, IndexError) as e:
        smb, msk, lon, lat, area = [np.array(data_nc[i]) for i in ["MBto", "MSK","LON","LAT", "AREA"]]
    smb = smb [:,0,:,:]
    m1  = (~ ((lat >= 75 ) & (lon <=-75))).astype(int)
    m1 *= (~ ((lat >=79.5) & (lon <=-67))).astype(int)
    m1 *= (~ ((lat >=81.2) & (lon <=-63))).astype(int)
    m1 *= (msk > 50).astype(int)
    m1 = m1.astype(float)
    m1*= msk/100.
    km3 = area / (1000*1000) 
    #plt.imshow(m1)
    #plt.show()
    return np.sum(np.sum(smb,axis=0)*m1*km3)
def smb_computation_from_annual_array(smb_anual, msk, lon, lat,area):
    # data_nc = Dataset(file)
    # try :
    #     smb, msk, lon, lat, area = [np.array(data_nc[i]) for i in ["MBTO", "MSK","LON","LAT", "AREA"]]
    # except( KeyError, IndexError) as e:
    #     smb, msk, lon, lat, area = [np.array(data_nc[i]) for i in ["MBto", "MSK","LON","LAT", "AREA"]]
    # smb = smb [:,0,:,:]
    m1  = (~ ((lat >= 75 ) & (lon <=-75))).astype(int)
    m1 *= (~ ((lat >=79.5) & (lon <=-67))).astype(int)
    m1 *= (~ ((lat >=81.2) & (lon <=-63))).astype(int)
    m1 *= (msk > 50).astype(int)
    m1 = m1.astype(float)
    m1*= msk/100.
    km3 = area / (1000*1000) 
    #plt.imshow(m1)
    #plt.show()
    return np.sum(smb_anual*m1*km3)
    # return np.sum(np.sum(smb,axis=0)*m1*km3)
