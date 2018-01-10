# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:27:52 2016
@author: eric orenstein

aspect-preserving image resizing
"""

import sys
sys.path.append('/home/aiswarya/caffe/python')

import numpy as np
import cv2
from skimage import morphology, restoration
from skimage.filters import scharr, gaussian
from scipy import ndimage

def aspect_resize(im, ii=226):
    """
    image == input array
    ii == desired dimensions
    """
    mm = [int(np.median(im[0, :, :])), int(np.median(im[1, :, :])), int(np.median(im[2, :, :]))]
    cen = np.floor(np.array((ii, ii))/2.0).astype('int')
    dim = im.shape[0:2]
    if dim[0] != dim[1]:
        # get the largest dimension
        large_dim = max(dim)
        
        # ratio between the large dimension and required dimension
        rat = float(ii)/large_dim
        
        # get the smaller dimension that maintains the aspect ratio
        small_dim = int(min(dim)*rat)
        
        # get the indicies of the large and small dimensions
        large_ind = dim.index(max(dim))
        small_ind = dim.index(min(dim))
        dim = list(dim)
        
        # the dimension assignment may seem weird cause of how python indexes images
        dim[small_ind] = ii
        dim[large_ind] = small_dim
        dim = tuple(dim)

        im = cv2.resize(im, dim)
        half = np.floor(np.array(im.shape[0:2])/2.0).astype('int')
        
        # make an empty array, and place the new image in the middle
        res = np.zeros((ii, ii, 3), dtype='uint8')
        res[:, :, 0] = mm[0]
        res[:, :, 1] = mm[1]
        res[:, :, 2] = mm[2]
        
        if large_ind == 1:
            test = res[cen[0]-half[0]:cen[0]+half[0], cen[1]-half[1]:cen[1]+half[1]+1]
            if test.shape != im.shape:
                res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]+1] = im
            else:
                res[cen[0]-half[0]:cen[0]+half[0], cen[1]-half[1]:cen[1]+half[1]+1] = im
        else:
            test = res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]]
            if test.shape != im.shape:
                res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]+1] = im
            else:
                res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]] = im
    else:
        res = cv2.resize(im, (ii, ii))
        half = np.floor(np.array(im.shape[0:2])/2.0).astype('int')

    return res
