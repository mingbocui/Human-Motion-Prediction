import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from scipy.interpolate import interp1d

def simpleTraj(fit_type, coeff):
    ##Observations 
    x_obs = np.tile(np.linspace(1, 8, num=8, endpoint=True), (7,1))
    y_obs = np.zeros((7, 8))
    ##Real Predictions
    x_pred = np.tile(np.linspace(9, 16, num=8, endpoint=True), (7,1))
    y_pred = np.zeros((7, 8))
    if fit_type == 'linear':
        y_pred[1, :] = 1*np.linspace(1, 8, num=8, endpoint=True)
        y_pred[2, :] = -1*np.linspace(1, 8, num=8, endpoint=True)
        y_pred[3, :] = 2*np.linspace(1, 8, num=8, endpoint=True)
        y_pred[4, :] = -2*np.linspace(1, 8, num=8, endpoint=True)
        y_pred[5, :] = 3*np.linspace(1, 8, num=8, endpoint=True)
        y_pred[6, :] = -3*np.linspace(1, 8, num=8, endpoint=True)
        
    if fit_type == 'square':
        xx = np.linspace(0, 7, num=8, endpoint=True)
        y_pred[1, :] = 3*np.power(xx,coeff)
        y_pred[2, :] = -3*np.power(xx, coeff)
        y_pred[3, :] = 5*np.power(xx,coeff)
        y_pred[4, :] = -5*np.power(xx,coeff)
        y_pred[5, :] = 7*np.power(xx,coeff)
        y_pred[6, :] = -7*np.power(xx,coeff)
        
    x = np.concatenate((x_obs, x_pred), axis=1)
    y = np.concatenate((y_obs, y_pred), axis=1)
    f1 = interp1d(x[0, :], y[0, :], kind='cubic')
    f2 = interp1d(x[1, :], y[1, :], kind='cubic')
    f3 = interp1d(x[2, :], y[2, :], kind='cubic')
    f4 = interp1d(x[3, :], y[3, :], kind='cubic')
    f5 = interp1d(x[4, :], y[4, :], kind='cubic')
    f6 = interp1d(x[5, :], y[5, :], kind='cubic')
    f7 = interp1d(x[6, :], y[6, :], kind='cubic')

    return (f1, f2, f3, f4, f5, f6, f7)

def createTrajDataset(filename, fitting_type='square', coeff=0.5, num_traj=1000, prob=[0.20, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00]):
    
    fs = simpleTraj(fitting_type, coeff)
    filename = filename
    num = num_traj
    prob = prob
    with open(filename, 'w') as the_file:
        frame_id = 0
        for i in range(num):
            ped_id = i
            ch = np.random.choice(7, 1, p=prob)
            ch = ch[0]
            x = np.linspace(1, 16, num=16, endpoint=True)

            import matplotlib.pyplot as plt
            f = fs[ch]
            y = f(x)
            y = np.around(y, 3)
            x = np.around(x, 3)
            plt.plot(x, y, 'r')
            plt.ylim((-10, 10))
            plt.xlim((0, 20))
            for j in range(16):
                frame_id += 1
                the_file.write(str(frame_id) + '\t' + str(ped_id) + '\t' + str(x[j]) + '\t' + str(y[j]) + '\n')
    plt.show()

    with open(filename) as f:
        content = f.readlines()
    f.close()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    print(content[0])
    print(str(content[1]))
