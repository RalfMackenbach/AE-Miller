import sys
sys.path.insert(1, '/Users/ralfmackenbach/Documents/GitHub/AE-tok/Miller/scripts')
import AE_tokamak_calculation as AEtok
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
def AE_func(omn,q,s,alpha,eta,epsilon=0.1,L_ref='minor'):
    if L_ref == 'minor':
        omn = omn/epsilon 
        int_val = np.asarray(quad(lambda k: integrand(k,s,alpha,q,omn,eta),  0, 1, limit=int(1e6)))
        return q**2 * epsilon**(5/2) * int_val
    if L_ref == 'major':
        return q**2 * epsilon**(1/2) * int_val


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)




# Construct grid for total integral
s_grid      =  np.linspace(-2.0, +2.0,   num=100 ,dtype='float64')
alpha_grid   = np.linspace(+0.0, +2.0,   num=100 ,dtype='float64')


sv, alphav     = np.meshgrid(s_grid, alpha_grid, indexing='ij')
AEv0            = np.empty_like(sv)
AEv1            = np.empty_like(sv)
AEv_err        = np.empty_like(sv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list0 = pool.starmap(AE_func, [(0.03, 2.0, sv[idx], alphav[idx], 0.0, 0.01, 'minor') for idx, val in np.ndenumerate(sv)])
    AE_list1 = pool.starmap(AEtok.calc_AE, [(0.03,0.0,0.01,2.0,1.0,0.0,0.0,sv[idx],0.0,0.0,alphav[idx],int(1e2),int(1e3),0.0,'minor') for idx, val in np.ndenumerate(sv)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list0 = np.asarray(AE_list0)
    AE_list1 = np.asarray(AE_list1)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(sv):
        AEv0[idx]    = AE_list0[list_idx][0]
        AEv1[idx]    = AE_list1[list_idx]
        list_idx = list_idx + 1

    rel_err = np.abs((AEv1-AEv0)/AEv0)

    levels0 = np.linspace(0, np.amax(AEv0), 25)
    levels1 = np.linspace(0, np.amax(AEv1), 25)
    levels2 = np.linspace(0, 0.1, 25)

    fig, axs = plt.subplots(1,3, figsize=(6.850394, 5.0/2)) #figsize=(6.850394, 3.0)
    cnt0 = axs[0].contourf(alphav, sv, AEv0, levels=levels0, cmap='plasma')
    cnt1 = axs[1].contourf(alphav, sv, AEv1, levels=levels1, cmap='plasma')
    cnt2 = axs[2].contourf(alphav, sv, rel_err, levels=levels2, cmap='plasma')
    for c in cnt0.collections:
        c.set_edgecolor("face")
    for c in cnt1.collections:
        c.set_edgecolor("face")
    for c in cnt2.collections:
        c.set_edgecolor("face")
    cbar0 = fig.colorbar(cnt0,ticks=[0.0, np.amax(AEv0)],ax=axs[0],orientation="horizontal",pad=0.2)
    cbar1 = fig.colorbar(cnt1,ticks=[0.0, np.amax(AEv1)],ax=axs[1],orientation="horizontal",pad=0.2)
    cbar2 = fig.colorbar(cnt2,ticks=[0.0, 0.1],ax=axs[2],orientation="horizontal",pad=0.2)
    cbar0.ax.set_xticklabels([r'$0$',r'$1.4 \times 10^{-4}$'])
    cbar1.ax.set_xticklabels([r'$0$',r'$1.4 \times 10^{-4}$'])
    cbar2.ax.set_xticklabels([r'$0\%$',r'$10\%$'])
    cbar0.solids.set_edgecolor("face")
    cbar1.solids.set_edgecolor("face")
    cbar2.solids.set_edgecolor("face")
    cbar0.set_label(r'$\widehat{A}$')
    cbar1.set_label(r'$\widehat{A}$')
    cbar2.set_label(r'$\mathrm{Error}$')
    axs[0].set_xlabel(r'$\alpha$')
    axs[1].set_xlabel(r'$\alpha$')
    axs[2].set_xlabel(r'$\alpha$')
    axs[0].set_ylabel(r'$s$')

    axs[0].text(1.7, -1.5, r'$(a)$',c='white',ha='center', va='center')
    axs[1].text(1.7, -1.5, r'$(b)$',c='white',ha='center', va='center')
    axs[2].text(1.7, -1.5, r'$(c)$',c='white',ha='center', va='center')

    # plt.text(3.2, -1.6, r'$(b)$',c='white')
    axs[0].xaxis.set_tick_params(which='major', direction='in', top='on')
    axs[0].xaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[0].yaxis.set_tick_params(which='major', direction='in', top='on')
    axs[0].yaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[1].xaxis.set_tick_params(which='major', direction='in', top='on')
    axs[1].xaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[1].yaxis.set_tick_params(which='major', direction='in', top='on')
    axs[1].yaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[2].xaxis.set_tick_params(which='major', direction='in', top='on')
    axs[2].xaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[2].yaxis.set_tick_params(which='major', direction='in', top='on')
    axs[2].yaxis.set_tick_params(which='minor', direction='in', top='on')
    plt.tight_layout()
    # plt.subplots_adjust(left=0.15, right=0.88, top=0.96, bottom=0.14)
    plt.margins(0.1)
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/s-alpha/code-comparison.eps', format='eps',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight', pad_inches = 0.01)
    plt.show()
