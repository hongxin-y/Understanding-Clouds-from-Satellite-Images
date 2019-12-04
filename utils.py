import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import keras
import cv2
from PIL import Image
from keras.applications.xception import Xception
from keras import backend as K
from keras import layers, optimizers
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
import os

def rle2mask(mask_rle, size = (2100, 1400)):
    mask = mask_rle.split()
    starts = np.array(mask[::2], dtype = int) - 1
    lengths = np.array(mask[1::2], dtype = int)
    # here dtype
    img = np.zeros(size[0]*size[1], dtype = np.uint8)
    for s, l in zip(starts, lengths):
        img[s:s+l] = np.ones(l)
    return img.reshape(size).T

def mask2rle(mask, size = (525, 350)):
    serial = mask.T.flatten()
    serial = np.concatenate([[0], serial, [0]])
    res = np.where(serial[1:] != serial[:-1])[0] + 1
    res[1::2] -= res[::2]
    return " ".join(list(map(str, res)))


def mask2contour(mask, width=5):
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2 = np.logical_xor(mask,mask2)
    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3 = np.logical_xor(mask,mask3)
    return np.logical_or(mask2,mask3)

def dice_coef(y_rle_true, y_rle_pred, probs, th):
    # there is not such label in the figure
    if probs < th:
        if y_rle_true=='': return 1
        else: return 0
    y_mask_true = rle2mask(y_rle_true)[::4,::4]
    y_mask_pred = np.array(Image.fromarray(rle2mask(y_rle_pred,size=(525,350))))
    union = np.sum(y_mask_true) + np.sum(y_mask_pred)
    # there are not union part in two figures
    if union==0: 
        return 1
    intersection = np.sum(y_mask_true * y_mask_pred)
    return 2. * intersection / union

    