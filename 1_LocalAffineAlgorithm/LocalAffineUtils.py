# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:58:03 2018

@author: WangZhao
pyhton3.6
Windows10
"""

import numpy as np
#平移向量======================================================================
def affine_trans(U,V):
    '''
    Input:U为猩猩特征点，V为人脸特征点
    Output:从U变换到V所需平移的向量
    '''
    trans = list(map(lambda x: x[0]-x[1], zip(V, U)))
    return trans

#双线性插值====================================================================
def Bilinear_interpolation(X,man_image):
    '''
    Input:图片上一点的坐标
    Output:坐标点的像素值
    '''
    #图片Size
    height, width, channels = man_image.shape   
    y, x = X[0], X[1]
    #取坐标点的整数部分，用于插值
    i, j = int(x), int(y)
    #取坐标点的小数部分，用于插值
    u, v = x-i, y-j
    #定义RGB
    RGB = np.zeros(channels, np.float)
    for k in range(channels):
        RGB[k] = (1-v)*(1-u)*man_image[j,i][k] + (1-v)*u*man_image[j,i+1][k] + v*(1-u)*man_image[j+1,i][k] + v*u*man_image[j+1,i+1][k]
    return RGB

