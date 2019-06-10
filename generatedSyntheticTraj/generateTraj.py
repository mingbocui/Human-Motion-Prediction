import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
from my_data import simpleTraj
from my_data import createTrajDataset
#%matplotlib inline

createTrajDataset('3Traj.txt', num_traj=500, prob = [1/3, 1/3, 1/3,0, 0, 0, 0])