import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

from MlsUtils import *

#MLS deformation function=======================================================
def MLS(manface,ape,points_q,points_p,alpha):
    trans_img = np.zeros(ape.shape)
    height, width, channels = ape.shape
    for y in range(height):
        for x in range(width):
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
            trans_img[y][x] = Bilinear_interpolation(f_v,manface)
            #fv = sum([A[j].dot(q_hat[j].reshape(2,1)) for j in range(A.shape[0])]).reshape(2,) + q_star
    return trans_img
