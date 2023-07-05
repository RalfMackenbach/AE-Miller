import AEtok.AE_tokamak_calculation as AEtok
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import h5py
import matplotlib        as mpl


omn     = "scan"
eta     = 1.0
q       = 2.0
s_q     = 0.0
alpha   = 0.0
dR0dr   = 0.0
s_kappa = 0.0
s_delta = 0.0
alpha   = 0.0
theta_res   = int(1e3+1)
L_ref       = 'minor'
epsilon = 1/3
rho = 0.5
Aspect_ratio = rho/epsilon


res = 20


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)




# Construct grid for total integral
kappa_grid          =  np.linspace(+0.5, +2.0, num=res)
delta_grid          =  np.linspace(-3/4, +3/4, num=res)
kappav, deltav      = np.meshgrid(kappa_grid,delta_grid)
AEv_kd              = np.empty_like(kappav)
AEv_kd_2            = np.empty_like(kappav)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AEkd_list =   pool.starmap(AEtok.calc_AE, [(2.0,eta,None,q,kappav[idx],deltav[idx],dR0dr,s_q,  s_kappa,s_delta,alpha,theta_res,L_ref,Aspect_ratio,rho) for idx, _ in np.ndenumerate(AEv_kd)])
    AEkd_list_2 = pool.starmap(AEtok.calc_AE, [(4.0,eta,None,q,kappav[idx],deltav[idx],dR0dr,s_q,  s_kappa,s_delta,alpha,theta_res,L_ref,Aspect_ratio,rho) for idx, _ in np.ndenumerate(AEv_kd)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()
    AEkd_list = np.asarray(AEkd_list)
    AEkd_list_2= np.asarray(AEkd_list_2)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv_kd):
        AEv_kd[idx]     = AEkd_list[list_idx]
        AEv_kd_2[idx]   = AEkd_list_2[list_idx]
        list_idx        = list_idx + 1


    hf      = h5py.File("/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/TGLF-comparison//data/AE_data_kd_eta={}.hdf5".format(eta), "w")
    hf.create_dataset('AEkd',       data=AEv_kd)
    hf.create_dataset('AEkd_2',     data=AEv_kd_2)
    hf.create_dataset('kappa',      data=kappav)
    hf.create_dataset('delta',      data=deltav)
    hf.close()

    plt.pcolor(kappav, deltav, AEv_kd_2, cmap='plasma')
    plt.colorbar()
    plt.show()