import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math

from scipy.interpolate import interp1d

def generateTraj(fit_type='square', coeff=0.5, num=3):
    ##Observations 
    x_obs = np.tile(np.linspace(1, 8, num=8, endpoint=True), (num,1))
    y_obs = np.zeros((num, 8))
    number = num
    ##Real Predictions
    x_pred = np.tile(np.linspace(9, 16, num=8, endpoint=True), (num,1))
    y_pred = np.zeros((num, 8))
    if fit_type == 'linear':
        for i in range(1, num):
            y_pred[i, :] = math.pow(-1,i+1)*math.ceil(i/2)*np.linspace(1, 8, num=8, endpoint=True)
        
#         y_pred[1, :] = 1*np.linspace(1, 8, num=8, endpoint=True)
#         y_pred[2, :] = -1*np.linspace(1, 8, num=8, endpoint=True)
#         y_pred[3, :] = 2*np.linspace(1, 8, num=8, endpoint=True)
#         y_pred[4, :] = -2*np.linspace(1, 8, num=8, endpoint=True)
#         y_pred[5, :] = 3*np.linspace(1, 8, num=8, endpoint=True)
#         y_pred[6, :] = -3*np.linspace(1, 8, num=8, endpoint=True)
        
    if fit_type == 'square' and num != 5:
        xx = np.linspace(0, 7, num=8, endpoint=True)
        for i in range(1, num):
            y_pred[i, :] = math.pow(-1,i+1)*(math.ceil(i/2)+2)*np.power(xx, coeff)
#         y_pred[2, :] = -1*np.power(xx, coeff)
#         y_pred[3, :] = 2*np.power(xx,coeff)
#         y_pred[4, :] = -2*np.power(xx,coeff)
#         y_pred[5, :] = 3*np.power(xx,coeff)
#         y_pred[6, :] = -3*np.power(xx,coeff)
    if fit_type == 'square' and num==5:
        xx = np.linspace(0, 7, num=8, endpoint=True)
        y_pred[1, :] = 3*np.power(xx, coeff)
        y_pred[2, :] = -3*np.power(xx, coeff)
        y_pred[3, :] = 5*np.power(xx,coeff)
        y_pred[4, :] = -5*np.power(xx,coeff)
        
    
    return x_obs, y_obs, x_pred, y_pred, number
