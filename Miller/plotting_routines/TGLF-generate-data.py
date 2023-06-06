import AEtok.AE_tokamak_calculation as AEtok
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import h5py
import matplotlib        as mpl


omn     = 1.0
eta     = 1.0
epsilon = 1/3
q       = 2.0
s_q     = 2.0
kappa   = 2.0
delta   =-0.3
dR0dr   = 0.0
s_kappa = 0.0
s_delta = 0.0
alpha   = 0.0
theta_res   = int(1e3+1)
L_ref       = 'minor'
A           = 3.0
rho         = 0.5


res = 50


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)




# Construct grid for total integral
kappa_grid          =  np.linspace(+0.5, +2.0, num=res)
delta_grid          =  np.linspace(-0.8, +0.8, num=res)

shear_grid          =  np.linspace(-2.0, +2.0, num=res)
alpha_grid          =  np.linspace( 0.0, +1.0, num=res)



kappav, deltav      = np.meshgrid(kappa_grid,delta_grid)
shearv, alphav      = np.meshgrid(shear_grid,alpha_grid)
AEv_kd              = np.empty_like(kappav)
AEv_sa              = np.empty_like(shearv)
AEv_sa_2              = np.empty_like(shearv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AEkd_list = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,q,kappav[idx],    deltav[idx],dR0dr,s_q,          s_kappa,s_delta,alpha,      theta_res,L_ref,A,rho) for idx, val in np.ndenumerate(AEv_kd)])
    AEsa_list = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,q,kappa,          delta,      dR0dr,shearv[idx],  s_kappa,s_delta,alphav[idx],theta_res,L_ref,A,rho) for idx, val in np.ndenumerate(AEv_sa)])
    AEsa_list_2 = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,q,kappa,          delta,      dR0dr,shearv[idx],  s_kappa,s_delta,alphav[idx],theta_res,L_ref,A,rho) for idx, val in np.ndenumerate(AEv_sa)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AEkd_list = np.asarray(AEkd_list)
    AEsa_list = np.asarray(AEsa_list)
    AEsa_list_2= np.asarray(AEsa_list_2)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv_kd):
        AEv_kd[idx]     = AEkd_list[list_idx]
        AEv_sa[idx]     = AEsa_list[list_idx]
        AEv_sa_2[idx]   = AEsa_list[list_idx]
        list_idx        = list_idx + 1


    hf      = h5py.File("/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/TGLF-comparison/AE_data.hdf5", "w")
    hf.create_dataset('AEkd',       data=AEv_kd)
    hf.create_dataset('kappa',      data=kappav)
    hf.create_dataset('delta',      data=deltav)
    hf.create_dataset('AEsa',       data=AEv_sa)
    hf.create_dataset('shear',      data=shearv)
    hf.create_dataset('alpha',      data=alphav)
    hf.create_dataset('AEsa_2',       data=AEv_sa_2)
    hf.create_dataset('shear_2',      data=shearv)
    hf.create_dataset('alpha_2',      data=alphav)
    hf.close()
