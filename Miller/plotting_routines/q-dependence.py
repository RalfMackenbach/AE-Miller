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
from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

random.seed(10)



def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)



# Construct grid for total integral
q_grid        = np.linspace(0, 10, num=101, dtype='float64')
q_grid        = q_grid[1::]


AEv            = np.empty_like(q_grid)
AEv_err        = np.empty_like(q_grid)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    miller_idx=0
    n_miller=4

    while miller_idx<n_miller:

        omn     = 3.0
        res     = 100
        eta     = 0.0
        epsilon = random.uniform(0.1,  0.5)
        q       = 'scan'
        kappa   = random.uniform(+1.0,  2.0)
        delta   = random.uniform(-0.7, +0.7)
        dR0dr   = 0.0
        s_q     = random.uniform(-0.5, +0.5)
        s_kappa = 0.0
        s_delta = 0.0
        alpha   = 0.0
        theta_res   = int(1e2 +1)
        lam_res     = int(1e3)
        del_sign    = 0.0
        L_ref       = 'major'

        # time the full integral
        start_time = time.time()
        AE_list = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,q_grid[idx],kappa,delta,dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(q_grid)])
        print("data generated in       --- %s seconds ---" % (time.time() - start_time))

        miller_idx=miller_idx+1

        AE_list = np.asarray(AE_list)

        # reorder data full int
        list_idx = 0
        for idx, val in np.ndenumerate(AEv):
            AEv[idx]    = AE_list[list_idx]
            list_idx    = list_idx + 1

        plt.plot(q_grid,AE_list/(q_grid**2),next(linecycler))
        plt.ylabel(r'$\widehat{A}/q^2$')
        plt.xlabel(r'$q$')

    pool.close()

    plt.xlim(0.0, 10.0)
    plt.ylim(bottom=0.0)
    plt.show()

