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
q       = 2.0
kappa   = 'scan'
delta   = 'scan'
dR0dr   = 0.0
s_kappa = 0.0
s_delta = 0.0
alpha   = 0.0
theta_res   = int(1e3)
L_ref       = 'minor'
A           = 3.0
rho         = 1/3


res = 20


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${}$'.format(a)



# Construct grid for total integral
kappa_grid      =  np.linspace(+1.0, +4.0, num=res)
delta_grid      =  np.linspace(-0.75, +0.75, num=res)


kappav, deltav = np.meshgrid(kappa_grid, delta_grid, indexing='ij')
AEv_1          = np.empty_like(kappav)
AEv_2          = np.empty_like(kappav)
AEv_err        = np.empty_like(kappav)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list_1 = pool.starmap(AEtok.calc_AE, [(0.81,3.0,None,2.02,kappav[idx],deltav[idx],-.11,.34,0.1,0.29,.096,theta_res,L_ref,6.0,0.5) for idx, val in np.ndenumerate(kappav)])
    AE_list_2 = pool.starmap(AEtok.calc_AE, [(omn,eta,None,q,kappav[idx],deltav[idx],dR0dr,2.0,s_kappa,s_delta,alpha,theta_res,L_ref,A,rho) for idx, val in np.ndenumerate(kappav)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list_1 = np.asarray(AE_list_1)
    AE_list_2 = np.asarray(AE_list_2)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv_1):
        AEv_1[idx]  = AE_list_1[list_idx]
        AEv_2[idx]  = AE_list_2[list_idx]
        list_idx    = list_idx + 1


    AE_plot_1 = np.log10(AEv_1)
    AE_plot_2 = np.log10(AEv_2)
    AE_min = np.amin(AE_plot_1)
    AE_max = np.amax(AE_plot_1)
    print(AE_min,AE_max)



    levels = np.linspace(AE_min, AE_max, 50)
    # make two plots
    fig, ax = plt.subplots(1, 2, figsize=(6, 2.1), constrained_layout=True)

    # make contour plots
    cnt0 = ax[0].contourf(deltav, kappav, AE_plot_1, levels=levels, cmap='plasma')
    cnt1 = ax[1].contourf(kappav, deltav, AE_plot_2, levels=levels, cmap='plasma')

    # make colorbar for right plot
    cbar = fig.colorbar(cnt1, ax=ax.ravel().tolist(),format=ticker.FuncFormatter(fmt),label=r'$\log \widehat{A}$')


    # for c in cnt.collections:
    #     c.set_edgecolor("face")
    # cbar = plt.colorbar(ticks=[AE_min, AE_max])
    # cbar.set_label(r'$\log \widehat{A}$')
    # # cbar.set_label(r'$\hat{A}$')
    # cbar.solids.set_edgecolor("face")

    #plt.title(r'Available Energy as a function of geometry' '\n' r'$\omega_n$={}, $\eta$={}, $\epsilon$={}, $q$={}, $d R_0/ d r$={}, $s_q$={}, $s_\kappa$={}, $s_\delta$={}, $\alpha$={}'  '\n'.format(omn,eta,epsilon,q,dR0dr,s_q,s_kappa,s_delta,alpha))
    ax[0].set_xlabel(r'$\kappa$')
    ax[1].set_xlabel(r'$\kappa$')
    ax[0].set_ylabel(r'$\delta$')
    ax[1].set_ylabel(r'$\delta$')
    # plt.text(1.65, -0.4, r'$(d)$',c='white')
    ax[0].xaxis.set_tick_params(which='major', direction='in', top='on')
    ax[0].xaxis.set_tick_params(which='minor', direction='in', top='on')
    ax[0].yaxis.set_tick_params(which='major', direction='in', top='on')
    ax[0].yaxis.set_tick_params(which='minor', direction='in', top='on')
    ax[1].xaxis.set_tick_params(which='major', direction='in', top='on')
    ax[1].xaxis.set_tick_params(which='minor', direction='in', top='on')
    ax[1].yaxis.set_tick_params(which='major', direction='in', top='on')
    ax[1].yaxis.set_tick_params(which='minor', direction='in', top='on')
    # plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plot_geom_wn={}_eta={}_eps={}_q={}_sq={}_dR0dr={}_skappa={}_sdelta={}_alpha={}.eps'.format(omn,eta,epsilon,q,s_q,dR0dr,s_kappa,s_delta,alpha), format='eps',
    #             #This is recommendation for publication plots
    #             dpi=1000,
    #             # Plot will be occupy a maximum of available space
    #             bbox_inches='tight')
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/delta-kappa/geom.png', format='png',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    plt.show()
