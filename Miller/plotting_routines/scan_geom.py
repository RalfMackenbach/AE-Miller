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
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

omn     = 3.0
eta     = 0.0
epsilon = 1/3
q       = 2.0
kappa   = 'scan'
delta   = 'scan'
dR0dr   = 0.0
s_q     = 0.0
s_kappa = 0.0
s_delta = 0.0
alpha   = 0.0
theta_res   = int(1e3)
lam_res     = int(1e3)
del_sign    = 0.0


dR0dr_f = lambda x: - 2 * ( x**2.0 + 1)/(3 * x**2.0 + 1) * 0.1 + 0.5 * (x**2.0 - 1)/(3*x**2.0 + 1)
s_delta_f = lambda x: x/np.sqrt(1 - x**2.0)



def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)



# Construct grid for total integral
kappa_grid      =  np.linspace(+0.5, +2.0, num=20, dtype='float64')
delta_grid      =  np.linspace(-0.8, +0.8, num=20, dtype='float64')


kappav, deltav = np.meshgrid(kappa_grid, delta_grid, indexing='ij')
AEv            = np.empty_like(kappav)
AEv_err        = np.empty_like(kappav)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,q,kappav[idx],deltav[idx],dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,lam_res,del_sign) for idx, val in np.ndenumerate(kappav)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list = np.asarray(AE_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv):
        AEv[idx]    = AE_list[list_idx]
        list_idx    = list_idx + 1



    levels = np.linspace(0, np.amax(AE_list**(3/2)), 37)
    fig = plt.figure(figsize=(3.375, 2.3))
    ax  = fig.gca()
    cnt = plt.contourf(kappav, deltav, AEv**(3/2),levels=levels, cmap='plasma')
    for c in cnt.collections:
        c.set_edgecolor("face")
    cbar = plt.colorbar(ticks=[0.0, np.amax(AE_list**(3/2))])
    cbar.set_label(r'$\widehat{A}^{3/2}$')
    # cbar.set_label(r'$\hat{A}$')
    cbar.solids.set_edgecolor("face")

    #plt.title(r'Available Energy as a function of geometry' '\n' r'$\omega_n$={}, $\eta$={}, $\epsilon$={}, $q$={}, $d R_0/ d r$={}, $s_q$={}, $s_\kappa$={}, $s_\delta$={}, $\alpha$={}'  '\n'.format(omn,eta,epsilon,q,dR0dr,s_q,s_kappa,s_delta,alpha))
    plt.xlabel(r'$\kappa$')
    plt.ylabel(r'$\delta$')
    # plt.text(1.65, -0.4, r'$(d)$',c='white')
    ax.xaxis.set_tick_params(which='major', direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', direction='in', top='on')
    ax.yaxis.set_tick_params(which='minor', direction='in', top='on')
    plt.tight_layout()
    # plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plot_geom_wn={}_eta={}_eps={}_q={}_sq={}_dR0dr={}_skappa={}_sdelta={}_alpha={}.eps'.format(omn,eta,epsilon,q,s_q,dR0dr,s_kappa,s_delta,alpha), format='eps',
    #             #This is recommendation for publication plots
    #             dpi=1000,
    #             # Plot will be occupy a maximum of available space
    #             bbox_inches='tight')
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/delta-kappa/geom.eps', format='eps',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    plt.show()
