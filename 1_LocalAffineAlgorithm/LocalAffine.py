# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:53:31 2018

@author: WangZhao
Python3.6
Windows10
"""

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import base64
import json
import cv2

from LocalAffineUtils import *

#利用旷世face++得到人脸特征点===================================================
import urllib.request
import urllib.error
import time
def get_man_points(man_path):
    '''
    以下代码参考旷视Face++API文档：https://console.faceplusplus.com.cn/documents/6329752 
    分别选取12，14，16个点进行比较
    '''
    http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    key = "RuF7pDgXRAl0EfZjZrWIYElt_QoO-XhN"
    secret = "lpUhYiPv8dSdFgMyjkaBKvO2X1nbKzv1"
    filepath = man_path
    #打开人脸图片
    fr = open(filepath,"rb")
    b64str = base64.b64encode(fr.read())
    fr.close()
    # API接口
    para={'api_key':key,'api_secret':secret,'image_base64':b64str,'return_landmark':1}
    DATA=urllib.parse.urlencode(para).encode("utf-8")
    # build http request
    req= urllib.request.urlopen(url=http_url, data=DATA)
    faceData=json.loads(req.read())
    #基准点
    man_points={}
    #'''眼睛'''----------------------------------------------------------------
    #左眼左眼角
    man_points['left_eye_left_corner']=faceData['faces'][0]['landmark']['left_eye_left_corner']
    #左眼右眼角
    man_points['left_eye_right_corner']=faceData['faces'][0]['landmark']['left_eye_right_corner']
    #左眼中心点
    man_points['left_eye_center']=faceData['faces'][0]['landmark']['left_eye_center']
    #man_points['left_eye_upper_right_quarter']=faceData['faces'][0]['landmark']['left_eye_upper_right_quarter']
    #右眼位置同左眼
    man_points['right_eye_left_corner']=faceData['faces'][0]['landmark']['right_eye_left_corner']
    man_points['right_eye_right_corner']=faceData['faces'][0]['landmark']['right_eye_right_corner']
    man_points['right_eye_center']=faceData['faces'][0]['landmark']['right_eye_center']
    #man_points['right_eye_upper_right_quarter']=faceData['faces'][0]['landmark']['right_eye_upper_right_quarter']
    #'''鼻子'''----------------------------------------------------------------
    #man_points['nose_contour_left1']=faceData['faces'][0]['landmark']['nose_contour_left1']
    #man_points['nose_contour_right1']=faceData['faces'][0]['landmark']['nose_contour_right1']
    #左鼻翼
    man_points['nose_left']=faceData['faces'][0]['landmark']['nose_left']
    #右鼻翼
    man_points['nose_right']=faceData['faces'][0]['landmark']['nose_right']
    #鼻尖
    man_points['nose_tip']=faceData['faces'][0]['landmark']['nose_tip']
    #'''嘴巴'''----------------------------------------------------------------
    man_points['mouth_left_corner']=faceData['faces'][0]['landmark']['mouth_left_corner']
    man_points['mouth_right_corner']=faceData['faces'][0]['landmark']['mouth_right_corner']
    
    man_points['mouth_lower_lip_bottom']=faceData['faces'][0]['landmark']['mouth_lower_lip_bottom']
    
    man_points['mouth_lower_lip_left_contour2']=faceData['faces'][0]['landmark']['mouth_lower_lip_left_contour2']
    man_points['mouth_lower_lip_right_contour2']=faceData['faces'][0]['landmark']['mouth_lower_lip_right_contour2']
    
    man_points['mouth_lower_lip_left_contour3']=faceData['faces'][0]['landmark']['mouth_lower_lip_left_contour3']
    man_points['mouth_lower_lip_right_contour3']=faceData['faces'][0]['landmark']['mouth_lower_lip_right_contour3']
        
    return man_points

#大猩猩特征点=======================================================================
def get_ape_points(height, width):
    '''
    click标注大猩猩特征点
    上限20个 
    包括：
        左右眼各4个特征点,共8个特征点
        鼻子5个特征点
        嘴巴7个特征点
    '''
    ape_points={}
    #'''眼睛'''----------------------------------------------------------------
    #左眼左眼角
    ape_points['left_eye_left_corner']={'y':0.103*height,'x':0.223*width}
    #左眼右眼角
    ape_points['left_eye_right_corner']={'y':0.147*height,'x':0.432*width}
    #左眼中心点
    ape_points['left_eye_center']={'y':0.114*height,'x':0.328*width}
    #左眼上1/4点
    ape_points['left_eye_upper_right_quarter']={'y':0.102*height,'x':0.428*width}
    #右眼左眼角
    ape_points['right_eye_left_corner']={'y':0.146*height,'x':0.549*width}
    #右眼右眼角
    ape_points['right_eye_right_corner']={'y':0.091*height,'x':0.752*width}
    #右眼中心点
    ape_points['right_eye_center']={'y':0.090*height,'x':0.652*width}
    #右眼上1/4点
    ape_points['right_eye_upper_right_quarter']={'y':0.102*height,'x':0.625*width}
    #'''鼻子''''---------------------------------------------------------------
    #鼻子左1
    ape_points['nose_contour_left1']={'y':0.148*height ,'x':0.465*width}
    #鼻子右1
    ape_points['nose_contour_right1']={'y':0.148*height,'x':0.528*width}
    #左鼻翼
    ape_points['nose_left']={'y':0.679*height ,'x':0.282*width}
    #右鼻翼
    ape_points['nose_right']={'y':0.681*height,'x':0.637*width}
    #鼻尖
    ape_points['nose_tip']={'y':0.720*height,'x':0.468*width}
    #'''嘴巴''''---------------------------------------------------------------
    #左嘴角
    ape_points['mouth_left_corner']={'y':0.762*height,'x':0.221*width}
    #右嘴角
    ape_points['mouth_right_corner']={'y':0.764*height,'x':0.759*width}
    #下唇底部
    ape_points['mouth_lower_lip_bottom']={'y':0.911*height,'x':0.474*width}
    #下唇左2
    ape_points['mouth_lower_lip_left_contour2']={'y':0.824*height,'x':0.271*width}
    #下唇右2
    ape_points['mouth_lower_lip_right_contour2']={'y':0.824*height,'x':0.713*width}
    #下唇左3
    ape_points['mouth_lower_lip_left_contour3']={'y':0.887*height,'x':0.357*width}
    #下唇右3
    ape_points['mouth_lower_lip_right_contour3']={'y':0.887*height,'x':0.609*width}
    #return特征点
    return ape_points

#局部仿射变换函数===============================================================    
def local_affine_trans(X,U,V,e):
    '''
    Input:第三张图（即变换后的图）上的点，大猩猩的仿射变换点，人脸特征点，参数e
    Output:变换后点的坐标
    '''
    #若X点为仿射变换点，则返回G(X)
    for p in U: 
        if (X[0] == U[p]['y'])&(X[1] == U[p]['x']):
            vector_b = affine_trans((U[p]['y'],U[p]['x']), (V[p]['y'],V[p]['x']))
            G = list(map(lambda x: x[0]+x[1], zip(X, vector_b)))
            return G
    #若X点不是仿射变换点，则利用距离进行加权求和
    Trans_X  = [0,0]
    #欧式距离的指数次方求和
    distance_sum = sum([1.0/((U[p]['y']-X[0])**2+(U[p]['x']-X[1])**2)**(0.5*e) for p in U])
    for p in U:
        #函数在LocalAffineUtils.py
        vector_b = affine_trans((U[p]['y'],U[p]['x']), (V[p]['y'],V[p]['x']))
        G = list(map(lambda x: x[0]+x[1], zip(X, vector_b)))
        distance = ((U[p]['y']-X[0])**2+(U[p]['x']-X[1])**2)**0.5
        w = (1.0/distance**e) / distance_sum
        w_G = list(map(lambda x: w*x, G))
        Trans_X = list(map(lambda x: x[0]+x[1], zip(Trans_X , w_G)))
    return Trans_X 

#定义形变函数===================================================================
def deformation(height, width,U,V,e,face,ape_face_new):
    # 对于狒狒图像的每一个像素点
    for y in range(height):
        for x in range(width):
            #利用仿射变换计算 大猩猩图片点所对应的人脸坐标
            T_X = local_affine_trans((y,x),U,V,e) 
            #双线性插值得出人脸该点像素值，插值函数在LocalAffineUtils.py
            RGB = Bilinear_interpolation(T_X,face) 
            #将算出的特定坐标下对应的像素值赋予大猩猩图片
            ape_face_new[y,x] = RGB #将狒狒图像像素点的RGB值进行替换 
    plt.imshow(ape_face_new)
    #隐去坐标轴
    plt.axis('off')

#==============================================================================
if __name__=='__main__':
    #给出人脸图片路径
    #man_path = "./data/gaoxiaosong.jpg"
    #man_path = "./data/zxh.jpg"
    man_path = "./data/wuyifan.jpg"
    #给出大猩猩图片路径
    ape_path = "./data/ape.jpg"
    #打开人脸图片
    man_face = Image.open(man_path)
    #人脸图片像素转化为数组
    man_face_data = np.array(man_face.convert('RGB'))
    #打开大猩猩图片
    ape_face = Image.open(ape_path)
    #大猩猩图片像素转化为数组
    ape_data = np.array(ape_face.convert('RGB'))
    #大猩猩图片Size
    height, width, channels = ape_data.shape
    #该图像用于形变图片像素填充
    ape_face_new = cv2.imread(ape_path,1)
    #得到人脸特征点
    V = get_man_points(man_path)
    #得到大猩猩特征点
    U = get_ape_points(height, width)
    #过滤掉Ape points有而 man points没有的点
    dictV_key=list(V.keys())
    dictU_key=list(U.keys())
    for key in dictU_key:
        if key not in dictV_key:
            U.pop(key, None)
    #print(U)
    #选定参数值e    
    e=2
    #e=3
    #e=1.5
    #结果
    Result = deformation(height, width,U,V,e,man_face_data,ape_face_new)


