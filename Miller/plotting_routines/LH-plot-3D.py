import AEtok.AE_tokamak_calculation as AEtok
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import h5py
import matplotlib        as mpl


eta     = 0.0
epsilon = 1/3
q       = 3.0
kappa   = 1.5
delta   =+0.5
dR0dr   =-0.5
s_kappa = 0.5
s_delta = delta/np.sqrt(1-delta**2)
theta_res   = int(1e3+1)
L_ref       = 'major'


res = 30


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)




# Construct grid for total integral
omn_grid          =  np.linspace(+0.1,+20.0, num=res, dtype='float64')
s_grid            =  np.linspace(+4.0, +3.0, num=res, dtype='float64')
alpha_grid        =  np.linspace(+0.0, +2.0, num=res, dtype='float64')


omnv, sv, alphav  = np.meshgrid(omn_grid,s_grid,alpha_grid)
AEv               = np.empty_like(omnv)
AEv_err           = np.empty_like(omnv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list = pool.starmap(AEtok.calc_AE, [(omnv[idx],eta,epsilon,q,kappa,delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,L_ref) for idx, val in np.ndenumerate(AEv)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list = np.asarray(AE_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv):
        AEv[idx]    = AE_list[list_idx]
        list_idx    = list_idx + 1


    hf      = h5py.File("/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/LH/isocontour_eta={}_eps={}_q={}_kappa={}_delta={}_dR0dr={}_skappa={}_sdelta={}.hdf5".format(eta,epsilon,q,kappa,delta,dR0dr,s_kappa,s_delta), "w")
    hf.create_dataset('omnv',   data=omnv)
    hf.create_dataset('sv',     data=sv)
    hf.create_dataset('alphav', data=alphav)
    hf.create_dataset('AEv',    data=AEv)
    hf.close()
