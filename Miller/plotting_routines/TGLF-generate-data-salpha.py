import AEtok.AE_tokamak_calculation as AEtok
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import h5py
import matplotlib        as mpl


omn     = 4.0
eta     = 1.0
q       = 2.0
s_q     = 0.0
alpha   = 0.0
dR0dr   = 0.0
s_kappa = 0.0
s_delta = 0.0
alpha   = 0.0
kappa  = 1.0
delta   = 0.0
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
gamma = 1.096665156462166


shear_grid          =  np.linspace(-1.0, +5.0, num=res)
alpha_grid          =  np.linspace( 0.0, +2.0/gamma, num=res)
shearv, alphav      = np.meshgrid(shear_grid,alpha_grid)
AEv_sa              = np.empty_like(shearv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AEsa_list =   pool.starmap(AEtok.calc_AE, [(4.0,eta,None,q,kappa,          delta,      dR0dr,shearv[idx],  s_kappa,s_delta,alphav[idx],theta_res,L_ref,Aspect_ratio,rho,'trapz') for idx, _ in np.ndenumerate(AEv_sa)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()
    AEsa_list = np.asarray(AEsa_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv_sa):
        AEv_sa[idx]     = AEsa_list[list_idx]
        list_idx        = list_idx + 1


    hf      = h5py.File("/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/TGLF-comparison/data/AE_data_salpha_eta={}.hdf5".format(eta), "w")
    hf.create_dataset('shear',      data=shearv)
    hf.create_dataset('alpha',      data=alphav)
    hf.create_dataset('AEsa',       data=AEv_sa)
    hf.close()

    plt.pcolor(alphav, shearv, AEv_sa, cmap='plasma')
    plt.colorbar()
    plt.show()