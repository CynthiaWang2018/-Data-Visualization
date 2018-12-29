# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 10:58:03 2018

@author: WangZhao
python3.6
windows10

"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

from MlsUtils import *

#MLS deformation function=======================================================
def MLS(manface,ape,points_q,points_p,alpha):
    '''
    Input:p为ape图像上的控制点
          q为man_face图像上的变换点
    Output:f(v)
           f(v)=q_i,if v=p_i,i=1,2,...,n  i.e.f(p_i)=q_i
           f(v)=..., o.w.
    '''
    trans_img = np.zeros(ape.shape)
    height, width, channels = ape.shape
    #如果v为ape图像上的控制点，有f(v)等于对应的q_i，反之用公式计算
    for y in range(height):
        for x in range(width):
            # 计算新的坐标
            v = np.array([y,x])
            idx = v_idx(v,points_p) 
            if idx>= 0:
                trans_img[y][x] = manface[points_q[idx][0]][[points_q[idx][1]]]
                continue
            #cite Image Deformation Using Moving Least Squares
            w = weight(points_p,v,alpha)
            w = w.reshape(w.shape[0],1)
            p_star = np.sum(w*points_p,0)/np.sum(w)
            q_star = np.sum(w*points_q,0)/np.sum(w)
            p_hat = points_p - p_star
            q_hat = points_p - q_star
            m = sum([w[i]*np.outer(p_hat[i].reshape(p_hat[i].shape[0],1),p_hat[i]) for i in range(len(w))])
            inv_m = inv(m)
            A = np.array([(v-p_star)*w[i]*inv_m.dot(p_hat[i].reshape(p_hat[i].shape[0],1)) for i in range(len(w))])
            f_v = sum([A[j].dot(q_hat[j].reshape(2,1)) for j in range(A.shape[0])]).reshape(2,) + q_star
            # 双线性插值得出像素值
            trans_img[y][x] = Bilinear_interpolation(f_v,manface)
            #fv = sum([A[j].dot(q_hat[j].reshape(2,1)) for j in range(A.shape[0])]).reshape(2,) + q_star
    return trans_img



#transformation function=======================================================
def transformation(manface_path,ape_path,alpha,Value,ape_points,manface_points):
    (manface,ape,points_q,points_p) = data(manface_path,ape_path,manface_points,ape_points,Value)
    trans = MLS(manface,ape,points_q,points_p,alpha)
    return trans

if __name__ == '__main__':
    '''
    #初始图片为庄老师-------------------------------------------------
    #人脸图片路径
    manface_path = './data/q1.jpg'
    #猩猩图片路径
    ape_path = './data/p1.jpg'
    #人脸特征点
    manface_points='./data/q_teacherZH.txt'
    #大猩猩的特征点
    ape_points='./data/p_points.txt'
    
   #初始图片为吴亦凡--------------------------------------------------
    #人脸图片路径
    manface_path = './data/wuyifan.jpg'
    #猩猩图片路径
    ape_path = './data/p1.jpg'
    #人脸特征点
    manface_points='./data/q_wuyifan.txt'
    #大猩猩的特征点
    ape_points='./data/p_points.txt'
    #Value控制点数 16个点
    Value ='./data/Value.txt'
    '''
   #图片为高晓松----------------------------------------------------
    
    #人脸图片路径
    manface_path = './data/gaoxiaosong.jpg'
    #猩猩图片路径
    ape_path = './data/p1.jpg'
    #人脸特征点
    manface_points='./data/q_gaoxiaosong.txt'
    #manface_points='./data/zxh.txt'
    #大猩猩的特征点
    ape_points='./data/p_points.txt'
    #ape_points='./data/ape.txt'
    #Value控制点数 16个点
    Value ='./data/Value.txt'
    
    #不同参数alpha下的mls变换结果
    i=0
    for alpha in [0.1,0.5,0.8,1]:
        i=i+1
        trans_img=transformation(manface_path,ape_path,alpha,Value,ape_points,manface_points)
        plt.subplot(1,4,i)
        plt.imshow(np.uint8(trans_img))
        plt.axis('off')
