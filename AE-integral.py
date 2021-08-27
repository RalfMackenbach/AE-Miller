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

from numba.extending import get_cython_function_address
from numba import vectorize, njit
import ctypes


Kaddr = get_cython_function_address("scipy.special.cython_special", "ellipk")
Eaddr = get_cython_function_address("scipy.special.cython_special", "ellipe")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
K_fn = functype(Kaddr)
E_fn = functype(Eaddr)


@vectorize('float64(float64)')
def vec_K(x):
    return K_fn(x)

@njit
def K_jit(x):
    return vec_K(x)

@vectorize('float64(float64)')
def vec_E(x):
    return E_fn(x)

@njit
def E_jit(x):
    return vec_E(x)





# @njit(error_model="numpy",fastmath=True)
# def integrand(z,k,wn,q,s,alpha,eta):
#     return (q**3 * np.exp(-z)* z**(1.5) * k * K_jit(k**2) *
#     np.maximum(0, +wn*(-0.5 - (2*alpha)/3. + (2*alpha*k**2)/3. - alpha/(4.*q**2)
#     - 2*s + 2*k**2*s + (1 - (2*alpha*(-1 + 2*k**2))/3. + 2*s)*
#     (E_jit(k**2)/K_jit(k**2)))*(1 + eta *
#     (z - 3/2))/(2) - z*(-0.5 - (2*alpha)/3. + (2*alpha*k**2)/3. -
#     alpha/(4.*q**2) - 2*s + 2*k**2*s + (1 - (2*alpha*(-1 + 2*k**2))/3. +
#     2*s)*(E_jit(k**2)/K_jit(k**2)))**2.0 ))


# # @nb.njit(error_model="numpy",fastmath=True,nopython=False)
# def AE_func(wn,q,s,beta,eta):
#     return dblquad(lambda z, k: integrand(z,k,wn,q,s,alpha,eta),  0, 1, lambda k: 0, lambda k: np.inf)








@njit(error_model="numpy",fastmath=True)
def integrand(z,k,wn,q,s,beta,eta):
    return (q**3 * np.exp(-z)* z**(1.5) * k * K_jit(k**2) *
    np.maximum(0, +wn*(-0.5 - (2*(q**2*beta*wn*(1+eta)))/3. + (2*(q**2*beta*wn*(1+eta))*k**2)/3. - (q**2*beta*wn*(1+eta))/(4.*q**2)
    - 2*s + 2*k**2*s + (1 - (2*(q**2*beta*wn*(1+eta))*(-1 + 2*k**2))/3. + 2*s)*
    (E_jit(k**2)/K_jit(k**2)))*(1 + eta *
    (z - 3/2))/(2) - z*(-0.5 - (2*(q**2*beta*wn*(1+eta)))/3. + (2*(q**2*beta*wn*(1+eta))*k**2)/3. -
    (q**2*beta*wn*(1+eta))/(4.*q**2) - 2*s + 2*k**2*s + (1 - (2*(q**2*beta*wn*(1+eta))*(-1 + 2*k**2))/3. +
    2*s)*(E_jit(k**2)/K_jit(k**2)))**2.0 ))

# @nb.njit(error_model="numpy",fastmath=True,nopython=False)
def AE_func(wn,q,s,beta,eta):
    return dblquad(lambda z, k: integrand(z,k,wn,q,s,beta,eta),  0, 1, lambda k: 0, lambda k: np.inf)






# Construct grid for total integral
wn_grid     = np.array([1.0,2.0,3.0,4.0]        ,dtype='float64')
q_grid      = np.array([0.5,1.0,1.5,2.0]        ,dtype='float64')
s_grid      = np.linspace(-4.0, +4.0,   num=101 ,dtype='float64')
beta_grid   = np.linspace(+0.0, +1.0,   num=101 ,dtype='float64')
eta_grid    = np.array([0.0,1.0,2.0,3.0]        ,dtype='float64')

wnv, qv, sv, betav, etav = np.meshgrid(wn_grid, q_grid, s_grid, beta_grid, eta_grid, indexing='ij')
AEv            = np.empty_like(wnv)
AEv_err        = np.empty_like(wnv)



if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print(mp.cpu_count())

    # time the full integral
    start_time = time.time()
    AE_list = pool.starmap(AE_func, [(wnv[idx], qv[idx], sv[idx], betav[idx], etav[idx]) for idx, val in np.ndenumerate(wnv)])
    print("total int took       --- %s seconds ---" % (time.time() - start_time))

    AE_list = np.asarray(AE_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(wnv):
        AEv[idx]    = AE_list[list_idx][0]
        AEv_err[idx]= AE_list[list_idx][1]
        list_idx = list_idx + 1


    # save data full int
    hf = h5py.File('data_full_beta_noprefac.h5', 'w')
    hf.create_dataset('wn_array',       data=wnv)
    hf.create_dataset('q_array',        data=qv)
    hf.create_dataset('shear_array',    data=sv)
    hf.create_dataset('beta_array',     data=betav)
    hf.create_dataset('eta_array',      data=etav)
    hf.create_dataset('AE_array',       data=AEv)
    hf.create_dataset('AE_error_array', data=AEv_err)
    hf.close()
