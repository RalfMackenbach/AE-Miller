import sys
sys.path.insert(1, '/Users/ralfmackenbach/Documents/GitHub/AE-tok/Miller/scripts')
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import h5py
import matplotlib        as mpl
import AE_tokamak_calculation as AEtok
from matplotlib import rc
import matplotlib.ticker as ticker
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

#             omn, eta, eps, q,   kappa, delta, dR0dr, s_q, s_kappa, s_delta, alpha
AEtok.calc_AE(3.0, 0.0, 0.8, 2.0, 0.7,   +0.5,  -0.1, -1.0, 0.0,     1.0,     0.0,theta_res=1000,lam_res=1000,del_sign=0.0,L_ref='major',rho=1.0,plot=True)


# for _, val in np.ndenumerate(np.logspace(-1,1,5)):
#     AEtok.calc_AE(3.0, 0.0, 1/3, val, 2.0,   -0.5,   0.0,   0.0, 0.0,     0.0,     1.0,theta_res=1000,lam_res=1000,del_sign=0.0,L_ref='major',rho=1.0,plot=True)