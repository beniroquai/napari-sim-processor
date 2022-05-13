#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 08:42:09 2022

@author: bene
"""

''' 
imports
'''

import numpy as np
import tifffile as tif

from napari_sim_processor.convSimProcessor import ConvSimProcessor
from napari_sim_processor.hexSimProcessor import HexSimProcessor

import matplotlib.pyplot as plt
#%%
'''
Auxilary functions
'''
def setReconstructor():
    '''
    Sets the attributes of the Processor
    Executed frequently, upon update of several settings
    '''
   
    h.usePhases = use_phases
    h.magnification = magnification
    h.NA = NA
    h.n = n
    h.wavelength = wavelength
    h.pixelsize = pixelsize
    h.alpha = alpha
    h.beta = beta
    h.w = w
    h.eta = eta
    if not find_carrier:
        h.kx = kx_input
        h.ky = ky_input
        


    
def get_current_stack_for_calibration(data):
    '''
    Returns the 4D raw image (angles,phases,y,x) stack at the z value selected in the viewer  
    '''
    if(0):
        data = np.expand_dims(np.expand_dims(data, 0), 0)
        dshape = data.shape # TODO: Hardcoded ...data.shape
        zidx = 0
        delta = group // 2
        remainer = group % 2
        zmin = max(zidx-delta,0)
        zmax = min(zidx+delta+remainer,dshape[2])
        new_delta = zmax-zmin
        data = data[...,zmin:zmax,:,:]
        phases_angles = phases_number*angles_number
        rdata = data.reshape(phases_angles, new_delta, dshape[-2],dshape[-1])            
        cal_stack = np.swapaxes(rdata, 0, 1).reshape((phases_angles * new_delta, dshape[-2],dshape[-1]))
    return data


#%%
'''
setup parameters
'''


mFile = "/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/MicronController/PYTHON/NAPARI-SIM-PROCESSOR/DATA/SIMdata_2019-11-05_15-21-42.tiff"
phases_number = 7
angles_number = 1
magnification = 60
NA = 1.05
n = 1.33
wavelength = 0.57
pixelsize = 6.5
dz= 0.55
alpha = 0.5
beta = 0.98
w = 0.2
eta = 0.65
group = 30
use_phases = True
find_carrier = True
phases_number = 7
pixelsize = 6.5
isCalibrated = False
use_phases =  True
use_torch = False
w = 0.2

#%%

'''
initialize
'''

# load images
mImages = tif.imread(mFile)


# set model
h = HexSimProcessor(); #ConvSimProcessor()
k_shape = (3,1)

# setup
h.debug = False
setReconstructor() 
kx_input = np.zeros(k_shape, dtype=np.single)
ky_input = np.zeros(k_shape, dtype=np.single)
p_input = np.zeros(k_shape, dtype=np.single)
ampl_input = np.zeros(k_shape, dtype=np.single)


#%%
'''
calibration
'''

imRaw = get_current_stack_for_calibration(mImages)         
if use_torch:
    h.calibrate_pytorch(imRaw,find_carrier)
else:
    h.calibrate(imRaw,find_carrier)
isCalibrated = True
if find_carrier: # store the value found   
    kx_input = h.kx  
    ky_input = h.ky
    p_input = h.p
    ampl_input = h.ampl 

'''
reconstruction
'''

assert isCalibrated, 'SIM processor not calibrated, unable to perform SIM reconstruction'
current_image = mImages#get_current_ap_stack()
dshape= current_image.shape
phases_angles = phases_number*angles_number
rdata = current_image.reshape(phases_angles, dshape[-2],dshape[-1])
if use_torch:
    imageSIM = h.reconstruct_pytorch(rdata.astype(np.float32)) #TODO:this is left after conversion from torch
else:
    imageSIM = h.reconstruct_rfftw(rdata)

plt.imshow(imageSIM), plt.show()


imname = 'SIM_' + imageRaw_name
show_image(imageSIM, im_name=imname, scale=[0.5,0.5], hold =True, autoscale = True)

