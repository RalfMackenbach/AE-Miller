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
from    scipy.signal        import  savgol_filter
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

omn     = 'scan'
res     = 100
eta     = 0.0
epsilon = 0.1
q       = 3.0
kappa   = 1.0
delta   = 0.0
dR0dr   = 0.0
s_q     = 0.0
s_kappa = 0.0
s_delta = 0.0
alpha   = 0.0
theta_res   = int(1e2 +1)
lam_res     = int(1e3)
del_sign    = 0.0
L_ref       = 'major'





def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)



# Construct grid for total integral
omn_grid        =  np.logspace(-2.0, +2.0, num=res, dtype='float64')


AEv            = np.empty_like(omn_grid)
AEv_err        = np.empty_like(omn_grid)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list = pool.starmap(AEtok.calc_AE, [(omn_grid[idx],eta,epsilon,q,kappa,delta,dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(omn_grid)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list = np.asarray(AE_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv):
        AEv[idx]    = AE_list[list_idx]
        list_idx    = list_idx + 1


    # plt.loglog(omn_grid,AE_list)
    # plt.loglog(omn_grid,AE_list[0]*(omn_grid/omn_grid[0])**(7/2),linestyle='dotted',color='black')
    # plt.loglog(omn_grid,AE_list[-1]*(omn_grid/omn_grid[-1])**(1),linestyle='dotted',color='black')
    # plt.axvline(x = crit_grad,color = 'r', linestyle='dashed')
    # plt.ylabel('available energy')
    # plt.xlabel('omn')
    plt.plot(omn_grid,AE_list)
    p=np.polyfit(omn_grid[-10::], AE_list[-10::], 1)
    plt.plot(omn_grid,p[0]*omn_grid + p[1],linestyle='dotted',color='black')
    print('crit grad is:', -p[1]/p[0])

    plt.ylim((0.0, AE_list[-1]/10))
    plt.xlim((0.0, 10.0))
    plt.show()
