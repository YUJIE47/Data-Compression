import numpy as np
import cv2

def DPCM(img):
    transform = []
    transform.append(int(img[0]))
    for i in range(1,len(img)):
        if i % 256 == 0:
            transform.append(int(img[i]) - int(img[i-256]))
        else:
            transform.append(int(img[i]) - int(img[i-1])) 
    transform = np.array(transform)
    return transform

def reverse_DPCM(img):
    tmp = []
    tmp.append(int(img[0]))
    for i in range(1,len(img)):
        if i % 256 == 0:
            tmp.append(int(img[i]) + int(tmp[i-256]))
        else:
            tmp.append(int(img[i]) + int(tmp[i-1])) 
    ntmp = np.array(tmp)
    return ntmp

