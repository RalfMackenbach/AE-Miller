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
import Miller_functions as Mf
import matplotlib.ticker as ticker
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

omn     = 3.0
eta     = 0.0
epsilon = 0.3
q       = 2.0
kappa   = 1.5
delta   = 0.6
dR0dr   = 0.0
s_q     = 'scan'
s_kappa = 0.0
s_delta = 0.0
alpha   = 'scan'
theta_res   = int(1e2 +1)
lam_res     = int(1e2)
del_sign    = 0.0




def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)



# Construct grid for total integral
s_grid          =  np.linspace(-2.0, +2.0, num=50, dtype='float64')
alpha_grid      =  np.linspace(+0.0, +2.0, num=50, dtype='float64')


sv, alphav = np.meshgrid(s_grid, alpha_grid, indexing='ij')
AEv            = np.empty_like(sv)
AEv_err        = np.empty_like(sv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,q,kappa,delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,lam_res,del_sign) for idx, val in np.ndenumerate(AEv)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list = np.asarray(AE_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv):
        AEv[idx]    = AE_list[list_idx]
        list_idx    = list_idx + 1

    np.save('AE_mat.npy', AEv)    # .npy extension is added if not given


    # theta = np.linspace(0,np.pi*2,theta_res)
    # f = plt.figure(1)
    # plt.plot(1 + epsilon*np.cos(theta + np.arcsin(delta)*np.sin(theta)), epsilon*kappa*np.sin(theta))
    # plt.axis('equal')
    # plt.title("Flux surface")
    # plt.xlabel(r'$R/R_0$')
    # plt.ylabel(r'$Z/R_0$')


    # levels = np.linspace(0, 4.0, 25)
    # print(np.amax(AEv))
    levels = np.linspace(0, np.amax(AE_list**(3/2)), 37)
    fig = plt.figure(figsize=(3.375, 2.3)) #figsize=(6.850394, 3.0)
    ax  = fig.gca()
    cnt = plt.contourf(alphav, sv, AEv**(3/2), levels=levels, cmap='plasma')
    for c in cnt.collections:
        c.set_edgecolor("face")
    cbar = plt.colorbar(ticks=[0.0, np.amax(AE_list**(3/2))])
    cbar.ax.set_yticklabels([r'$0.0$', r'$6.3$'])
    cbar.set_label(r'$\widehat{A}^{3/2}$')
    # cbar.set_label(r'$\widehat{A}$')
    cbar.solids.set_edgecolor("face")
    # plt.title(r'Available Energy as a function of $s$ and $\alpha$' '\n' r'$\omega_n$={}, $\eta$={}, $\epsilon$={}, $q$={}, $d R_0/ d r$={}, $\kappa$={}, $s_\kappa$={}, $\delta$={}, $s_\delta$={}' '\n'.format(omn,eta,epsilon,q,dR0dr,kappa,s_kappa,delta,s_delta))
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$s$')
    # plt.text(3.2/4, -1.6, r'$(a)$',c='white')
    ax.xaxis.set_tick_params(which='major', direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', direction='in', top='on')
    ax.yaxis.set_tick_params(which='minor', direction='in', top='on')
    plt.tight_layout()
    # plt.subplots_adjust(left=0.15, right=0.88, top=0.96, bottom=0.14)
    plt.margins(0.1)
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/s-alpha/s-alpha.eps'.format(omn,eta,epsilon,q,kappa,delta,dR0dr,s_kappa,s_delta), format='eps',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    plt.show()
