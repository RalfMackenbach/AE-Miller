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
import random
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

random.seed(99)



def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)



delta   = np.linspace(-0.8,0.8,50)


AEv            = np.empty_like(delta)
AEv_err        = np.empty_like(delta)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    miller_idx=0
    
    n_miller=6
    s_q_arr = np.linspace(0.0,1.0,n_miller)
    while miller_idx<n_miller:
    

        omn     = 1.0
        res     = 100
        eta     = 2.0
        epsilon = 0.3
        q       = 2.0
        kappa   = 1.5
        #delta   = 'scan'
        dR0dr   = 0.0
        s_q     = 1.0
        s_kappa = 0.0
        s_delta = 0.0
        alpha   = s_q_arr[miller_idx]
        theta_res   = int(1e2 +1)
        lam_res     = int(1e3)
        del_sign    = 0.0
        L_ref       = 'minor'


        # time the full integral
        start_time = time.time()
        AE_list = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,q,kappa,delta[idx],dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(delta)])
        print("data generated in       --- %s seconds ---" % (time.time() - start_time))

        miller_idx=miller_idx+1

        AE_list = np.asarray(AE_list)

        # reorder data full int
        list_idx = 0
        for idx, val in np.ndenumerate(AEv):
            AEv[idx]    = AE_list[list_idx]
            list_idx    = list_idx + 1

        print(np.amax(AEv))

        plt.plot(delta,AE_list/np.amax(AEv),label=alpha)
        plt.ylabel(r'$\widehat{A}$')
        plt.xlabel(r'$\delta$')
        plt.xlim(-0.8,0.8)
        plt.ylim(0.0, 1.0)

    pool.close()
    plt.legend()
    plt.show()
