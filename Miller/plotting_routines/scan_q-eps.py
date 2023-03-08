import AEtok.AE_tokamak_calculation as AEtok
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import matplotlib        as mpl
from matplotlib import rc
import matplotlib.ticker as ticker
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

omn     = 1.0
eta     = 0.0
epsilon = 'scan'
q       = 'scan'
kappa   = 1.0
delta   = 0.0
dR0dr   = 0.0
s_q     = 0.0
s_kappa = 0.0   
s_delta = 0.0
alpha   = 0.0
theta_res   = int(1e3+1)
L_ref       = 'minor'
A           = 3.0
rho         = 0.7



def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)



# Construct grid for total integral
eps_grid        =  np.logspace(-3,  -0.5,  num=100)
q_grid          =  np.logspace(-2,   1.0,  num=100)


epsv, qv = np.meshgrid(eps_grid, q_grid, indexing='ij')
AEv            = np.empty_like(epsv)
AEv_err        = np.empty_like(epsv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list = pool.starmap(AEtok.calc_AE, [(omn,eta,epsv[idx],qv[idx],kappa,delta,dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,L_ref,A,rho) for idx, val in np.ndenumerate(epsv)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list = np.asarray(AE_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv):
        AEv[idx]    = AE_list[list_idx]
        list_idx    = list_idx + 1



    plot = AEv / ( np.sqrt(epsv) )

    levels = np.linspace(0, np.amax(plot), 37)
    fig = plt.figure(figsize=(3.375, 2.3))
    ax  = fig.gca()
    cnt = plt.contourf(epsv, qv, plot,levels=levels, cmap='plasma')
    for c in cnt.collections:
        c.set_edgecolor("face")
    cbar = plt.colorbar(ticks=[0.0, np.amax(plot)])#np.amax(AE_list**(1.0))])
    cbar.set_label(r'$\widehat{A}/\sqrt{\epsilon} $')
    # cbar.set_label(r'$\hat{A}$')
    cbar.solids.set_edgecolor("face")

    #plt.title(r'Available Energy as a function of geometry' '\n' r'$\omega_n$={}, $\eta$={}, $\epsilon$={}, $q$={}, $d R_0/ d r$={}, $s_q$={}, $s_\kappa$={}, $s_\delta$={}, $\alpha$={}'  '\n'.format(omn,eta,epsilon,q,dR0dr,s_q,s_kappa,s_delta,alpha))
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$q$')
    # plt.text(1.65, -0.4, r'$(d)$',c='white')
    ax.xaxis.set_tick_params(which='major', direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', direction='in', top='on')
    ax.yaxis.set_tick_params(which='minor', direction='in', top='on')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    # plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plot_geom_wn={}_eta={}_eps={}_q={}_sq={}_dR0dr={}_skappa={}_sdelta={}_alpha={}.eps'.format(omn,eta,epsilon,q,s_q,dR0dr,s_kappa,s_delta,alpha), format='eps',
    #             #This is recommendation for publication plots
    #             dpi=1000,
    #             # Plot will be occupy a maximum of available space
    #             bbox_inches='tight')
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/delta-kappa/epsq.eps', format='eps',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    plt.show()

