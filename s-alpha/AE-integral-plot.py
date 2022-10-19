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
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
import matplotlib.ticker as ticker

from numba.extending import get_cython_function_address
from numba import vectorize, njit
import ctypes
import sys
wn_input = float(sys.argv[1])
q_input  = float(sys.argv[2])
eta_input= float(sys.argv[3])


Kaddr = get_cython_function_address("scipy.special.cython_special",    "ellipk")
Eaddr = get_cython_function_address("scipy.special.cython_special",    "ellipe")
erfaddr = get_cython_function_address("scipy.special.cython_special",  "__pyx_fuse_1erf")
erfcaddr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_0erfc")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
K_fn = functype(Kaddr)
E_fn = functype(Eaddr)
erf_fn = functype(erfaddr)
erfc_fn = functype(erfcaddr)


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

@vectorize('float64(float64)')
def vec_erf(x):
    return erf_fn(x)

@njit
def erf_jit(x):
    return vec_erf(x)

@vectorize('float64(float64)')
def vec_erfc(x):
    return erfc_fn(x)

@njit
def erfc_jit(x):
    return vec_erfc(x)



# wa = (-0.5 - (2*alpha)/3. + (2*alpha*k**2)/3. - alpha/(4.*q**2) - 2*s + 2*k**2*s + ((1 + (2*alpha)/3. - (4*alpha*k**2)/3. + 2*s)*E_jit(k**2))/K_jit(k**2))


@njit(error_model="numpy",fastmath=True)
def I_z(c0,c1):
    if (c1 <= 0) and (c0 >= 0):
        return 2 * c0 - 5 * c1
    if (c0 >= 0) and (c1 > 0):
        return (2*np.sqrt(c0/c1)*(4*c0 + 15*c1))/(3.*np.exp(c0/c1)*np.sqrt(np.pi)) + (2*c0 - 5*c1)*erf_jit(np.sqrt(c0/c1))
    if (c0 < 0) and (c1 < 0):
        return (-8*c0*np.sqrt(c0/c1) - 30*c0*np.sqrt(c1/c0) + 3*(2*c0 - 5*c1)*np.exp(c0/c1)*np.sqrt(np.pi)*erfc_jit(np.sqrt(c0/c1)))/(3.*np.exp(c0/c1)*np.sqrt(np.pi))
    else:
        return 0.0

@njit(error_model="numpy",fastmath=True)
def w_lam(k,s,alpha,q):
    return 2*(-0.5 - alpha/(4.*q**2) + 2*s*(-1 + k**2 + E_jit(k**2)/K_jit(k**2)) - (2*alpha*(1 - k**2 + ((-1 + 2*k**2)*E_jit(k**2))/K_jit(k**2)))/3. + E_jit(k**2)/K_jit(k**2))



@njit(error_model="numpy",fastmath=True)
def integrand(k,s,alpha,q,omn,eta):
    oml = w_lam(k,s,alpha,q)
    c0 = omn/oml*(1.0 - 3.0/2.0 * eta)
    c1 = 1.0 - omn/oml*eta
    return 4*np.sqrt(2)/np.pi*k*K_jit(k**2.0)*I_z(c0,c1)*oml**2.0


# @nb.njit(error_model="numpy",fastmath=True,nopython=False)
def AE_func(omn,q,s,alpha,eta,epsilon=0.3,L_ref='minor'):
    if L_ref == 'minor':
        omn = omn/epsilon 
        return q**2 * epsilon**(5/2) * quad(lambda k: integrand(k,s,alpha,q,omn,eta),  0, 1, limit=int(1e6))
    if L_ref == 'major':
        return q**2 * epsilon**(1/2) * quad(lambda k: integrand(k,s,alpha,q,omn,eta),  0, 1, limit=int(1e6), points=1)


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)




# Construct grid for total integral
s_grid      =  np.linspace(-2.0, +2.0,   num=100 ,dtype='float64')
alpha_grid   = np.linspace(+0.0, +4.0,   num=100 ,dtype='float64')


sv, alphav     = np.meshgrid(s_grid, alpha_grid, indexing='ij')
AEv            = np.empty_like(sv)
AEv_err        = np.empty_like(sv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list = pool.starmap(AE_func, [(wn_input, q_input, sv[idx], alphav[idx], eta_input) for idx, val in np.ndenumerate(sv)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list = np.asarray(AE_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(sv):
        AEv[idx]    = AE_list[list_idx][0]
        AEv_err[idx]= AE_list[list_idx][1]
        list_idx = list_idx + 1



    fig = plt.figure() #figsize=(3.375, 2.3)
    ax  = fig.gca()
    cnt = plt.contourf(alphav, sv, AEv, 25, cmap='plasma')
    for c in cnt.collections:
        c.set_edgecolor("face")
    cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
    cbar.set_label(r'$\widehat{A}^{3/2}$')
    cbar.solids.set_edgecolor("face")
    plt.title(r'Available Energy as a function of $s$ and $\alpha$' '\n' r'$\omega_n$={}, $\eta$={}, $q$={},' '\n'.format(wn_input,eta_input,q_input))
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$s$')
    # plt.text(3.2, -1.6, r'$(b)$',c='white')
    ax.xaxis.set_tick_params(which='major', direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', direction='in', top='on')
    ax.yaxis.set_tick_params(which='minor', direction='in', top='on')
    plt.tight_layout()
    # plt.subplots_adjust(left=0.15, right=0.88, top=0.96, bottom=0.14)
    plt.margins(0.1)
    plt.show()
