import sys
sys.path.insert(1, '/Users/ralfmackenbach/Documents/GitHub/AE-tok/Miller/scripts')
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import matplotlib        as mpl
import AE_tokamak_calculation as AEtok
from matplotlib import rc
import matplotlib.ticker as ticker
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

#                omn, eta,  eps,  q,   kappa, delta, dR0dr, s_q, s_kappa, s_delta, alpha
ae=AEtok.calc_AE(1e+2,1000,  1/3,  2.0, 3/2,  +0.5,   0.0,   0.0, 0.0,     0.0,     0.0,  theta_res=10001,lam_res=1000,L_ref='major',rho=1.0,int_meth='trapz',plot_precs=True,plot_dist=False)
print(ae)