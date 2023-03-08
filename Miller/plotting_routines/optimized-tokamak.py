import AEtok.AE_tokamak_calculation as AEtok
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import matplotlib        as mpl
from matplotlib import rc
import matplotlib.ticker as ticker
import scipy
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

omn     = 1/1e10
eta     = 1e10
epsilon = 1/3
q       = 2.0
kappa   = 1.0
delta   = 0.0
dR0dr   = 0.0
s_q     = 1.0
s_kappa = 0.0
s_delta = 0.0
alpha   = 1.0
theta_res   = int(1e3+1)
L_ref       = 'minor'
A       = 3.0 
rho     = 1.0
plot_bool = False



#                            (omn,eta,epsilon,  q,kappa,delta,dR0dr, s_q,s_kappa,s_delta,alpha,theta_res,lam_res,del_sign,L_ref,rho)
fun = lambda x: AEtok.calc_AE(omn,eta,epsilon,  q,x[0], x[1],dR0dr, s_q,s_kappa,s_delta,alpha,theta_res,L_ref,A,rho,plot_bool)
res = scipy.optimize.shgo(fun,bounds=((0.5,2.0),(-0.5,0.5)),options={'disp': True} )
vals=res.x
plot_bool = True
fun(vals)