# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 10:58:03 2018

@author: WangZhao
python3.6
windows10
"""
#from skimage import transform as tf
import numpy as np
from math import floor
from PIL import Image

#若v属于控制点,返回控制点的index=================================================
def v_idx(v,points_p):
    for i,p in enumerate(points_p):
        if (np.array(p)==v).all():
            return i
    return -1

#计算权重值====================================================================
def weight(points_p,v,alpha):
    return np.array([1.0/np.sum(np.abs(v-p)**(2*alpha)) for p in points_p])
'''
#双线性插值======================================================================
def Bilinear_interpolation(f_v,manface):
    height,width = manface.shape[0:2]
    i,j = f_v
    #得到整数部分，和小数部分
    i_0, u = int(floor(i)),i - floor(i)
    j_0, v = int(floor(j)),j - floor(j)
    i_1 = i_0+1
    j_1 = j_0+1
    if i_0 == height-1:
        i_1 = i_0
        u = 0
    if j == width-1:
        j_1 = j_0
        v = 0
    return (1-u)*(1-v)*manface[i_0][j_0]+(1-u)*v*manface[i_0][j_1]+u*(1-v)*manface[i_1][j_0] +u*v*manface[i_1][j_1]
'''
#Linear interpolation function
def Bilinear_interpolation(location,img):
    x_size,y_size = img.shape[0:2]
    ori_i,ori_j = location
    new_i, u = int(floor(ori_i)),ori_i - floor(ori_i)
    new_j, v = int(floor(ori_j)),ori_j - floor(ori_j)
    #adjust when it is the border 
    new_ii = new_i+1
    new_jj = new_j+1
    if new_i == x_size-1:
        new_ii = new_i
        u = 0
    if new_j == y_size-1:
        new_jj = new_j
        v = 0
    #interpolation assignment
    return (1-u)*(1-v)*img[new_i][new_j]+(1-u)*v*img[new_i][new_jj]+u*(1-v)*img[new_ii][new_j] +u*v*img[new_ii][new_jj]

#导入数据======================================================================
def load_data(path):
    points = {}
    with open(path,'r') as reader:
        for line in reader:
            key,value = line.strip().split('=')
            x,y = value.split(',')
            x,y = int(x),int(y)
            points[key] = np.array([y,x])
    return points

#导入数据===================================================================
def load_value(path):
    lines = []
    with open(path,'r') as reader:
        for line in reader:
            terms = line.strip().split(',')
            lines.append(terms)
    return lines 

#=======================================================================
def data(manface_path,ape_path,manface_data,ape_data,regions_data):
    manface = Image.open(manface_path)
    manface = np.array(manface) 
    ape = Image.open(ape_path)
    ape= np.array(ape)
    #Load control points data
    ape_dict = load_data(ape_data)
    manface_dict = load_data(manface_data)
    #Load the regions data we want to use
    regions = load_value(regions_data)
    #Change the dictionary data to list data
    #人脸特征点
    points_q = []
    #大猩猩的特征点
    points_p = []
    for i,keys in enumerate(regions):
        p = []
        q = []
        for key in keys:
            if key not in manface_dict or key not in ape_dict:
                return ('The control regions choose is not correct,please change it'),None
            p.append(ape_dict[key])
            q.append(manface_dict[key])
        
        if len(keys) == 1:
            points_p.append(p[0])
            points_q.append(q[0])            
    return (manface,ape,points_q,points_p) 
