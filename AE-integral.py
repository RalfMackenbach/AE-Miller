import scipy
import scipy.special
from   scipy import integrate
import numpy as np
from   scipy.integrate import dblquad, quad
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import h5py
import matplotlib        as mpl


def ramp(x):
    return np.maximum(0, x)

def AE_func(omn,s,eta):
    return dblquad(lambda z, k: np.exp(-z)* z**(2.5) * k * \
    (((2 + 4 * s) * scipy.special.ellipe(k**2.0) + (-1 + 4 * (-1 + k**2.0) * s)
    * scipy.special.ellipk(k**2.0))**2.0) / ( scipy.special.ellipk(k**2.0) ) *
    ramp( (-4 * (z + 2 * s * z) * scipy.special.ellipe(k**2.0) +
    (2 * z - 8 * (-1 + k**2.0) * s * z + omn * (-3 + 2 * z + 2 * eta)) *
    scipy.special.ellipk(k**2.0))/( 4 * (z + 2 * s * z) *
    scipy.special.ellipe(k**2.0) + 2 * (-1 + 4 * (-1 + k**2.0) * s)
    * z * scipy.special.ellipk(k**2.0)) ),  0, 1, lambda k: 0, lambda k: np.inf)

def AE_func_trapped(k,omn,s,eta):
    return quad(lambda z: np.exp(-z)* z**(2.5) * k * \
    (((2 + 4 * s) * scipy.special.ellipe(k**2.0) + (-1 + 4 * (-1 + k**2.0) * s)
    * scipy.special.ellipk(k**2.0))**2.0) / ( scipy.special.ellipk(k**2.0) ) *
    ramp( (-4 * (z + 2 * s * z) * scipy.special.ellipe(k**2.0) +
    (2 * z - 8 * (-1 + k**2.0) * s * z + omn * (-3 + 2 * z + 2 * eta)) *
    scipy.special.ellipk(k**2.0))/( 4 * (z + 2 * s * z) *
    scipy.special.ellipe(k**2.0) + 2 * (-1 + 4 * (-1 + k**2.0) * s)
    * z * scipy.special.ellipk(k**2.0)) ), 0, np.inf)




# Construct grid for total integral
omn_grid = np.linspace(+0.0, +4.0, num=100)
s_grid   = np.linspace(-2.0, +2.0, num=100)
eta_grid = np.linspace(+0.0, +2.0, num=4)
omnv, sv, etav = np.meshgrid(omn_grid, s_grid, eta_grid, indexing='ij')
AEv            = np.empty_like(omnv)
AEv_err        = np.empty_like(omnv)


#construct grid for integral over energy
omn_grid    = np.linspace(+1.0, +4.0,   num=4)
k_grid      = np.linspace(+0.0, +1.0,   num=1000)
s_grid      = np.linspace(-2.0, +2.0,   num=1000)
kv_z, omnv_z, sv_z = np.meshgrid(k_grid, omn_grid, s_grid, indexing='ij')
AEv_z            = np.empty_like(omnv_z)
AEv_z_err        = np.empty_like(omnv_z)

if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print(mp.cpu_count())

    # # time the full integral
    # start_time = time.time()
    # AE_list = pool.starmap(AE_func, [(omnv[idx], sv[idx], etav[idx]) for idx, val in np.ndenumerate(omnv)])
    # print("total int took       --- %s seconds ---" % (time.time() - start_time))


    # time the partial integral
    start_time = time.time()
    AE_z_list = pool.starmap(AE_func_trapped, [(kv_z[idx],omnv_z[idx], sv_z[idx], 2/3) for idx, val in np.ndenumerate(omnv_z)])
    print("partial int took     --- %s seconds ---" % (time.time() - start_time))
    pool.close()

    ## reorder data full int
    # list_idx = 0
    # for idx, val in np.ndenumerate(omnv):
    #     AEv[idx]    = AE_list[list_idx][0]
    #     AEv_err[idx]= AE_list[list_idx][1]
    #     list_idx = list_idx + 1

    ## reorder data partial int
    list_idx = 0
    for idx, val in np.ndenumerate(omnv_z):
        AEv_z[idx]    = AE_z_list[list_idx][0]
        AEv_z_err[idx]= AE_z_list[list_idx][1]
        list_idx = list_idx + 1

    ## save data full int
    # hf = h5py.File('data_full.h5', 'w')
    # hf.create_dataset('omn_array',      data=omnv)
    # hf.create_dataset('shear_array',    data=sv)
    # hf.create_dataset('eta_array',      data=etav)
    # hf.create_dataset('AE_array',       data=AEv)
    # hf.create_dataset('AE_error_array', data=AEv_err)
    # hf.close()

    # save data partial int
    hp = h5py.File('data_partial.h5', 'w')
    hp.create_dataset('k_array',        data=kv_z)
    hp.create_dataset('omn_array',         data=omnv_z)
    hp.create_dataset('shear_array',    data=sv_z)
    hp.create_dataset('AE_array',       data=AEv_z)
    hp.create_dataset('AE_error_array', data=AEv_z_err)
    hp.close()
