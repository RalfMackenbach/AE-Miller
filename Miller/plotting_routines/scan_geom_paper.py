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
epsilon = 1/3
q       = 2.0
kappa   = 3/2
delta   = 0.0
dR0dr   = 0.0
s_q     = 1.0
s_kappa = 0.0
s_delta = 0.0
alpha   = 0.0
theta_res   = int(1e3 +1)
L_ref       = 'major'




def fmt(x, pos):
    a = round(x,pos)
    return r'${}$'.format(a)


res = 10


# Construct grid for total integral
kappa_grid      =  np.linspace(+0.5, +2.0, num=res)
delta_grid      =  np.linspace(-0.8, +0.8, num=res)


kappav, deltav = np.meshgrid(kappa_grid, delta_grid, indexing='ij')
AEv_0   = np.empty_like(kappav)
AEv_1   = np.empty_like(kappav)
AEv_2   = np.empty_like(kappav)
AEv_3   = np.empty_like(kappav)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))


    # time the full integral
    start_time = time.time()
    AE_list_0 = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,  q, kappav[idx], deltav[idx],dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,L_ref) for idx, val in np.ndenumerate(AEv_0)])

    AE_list_1 = pool.starmap(AEtok.calc_AE, [(omn,eta,    2/3,  q, kappav[idx], deltav[idx],dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,L_ref) for idx, val in np.ndenumerate(AEv_1)])

    AE_list_2 = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,  q, kappav[idx], deltav[idx],dR0dr,s_q,s_kappa,s_delta,0.5,  theta_res,L_ref) for idx, val in np.ndenumerate(AEv_2)])

    AE_list_3 = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,  q, kappav[idx], deltav[idx],dR0dr,0.0,s_kappa,s_delta,alpha,theta_res,L_ref) for idx, val in np.ndenumerate(AEv_3)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list_0 = np.asarray(AE_list_0)
    AE_list_1 = np.asarray(AE_list_1)
    AE_list_2 = np.asarray(AE_list_2)
    AE_list_3 = np.asarray(AE_list_3)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv_0):
        AEv_0[idx]  = AE_list_0[list_idx]
        AEv_1[idx]  = AE_list_1[list_idx]
        AEv_2[idx]  = AE_list_2[list_idx]
        AEv_3[idx]  = AE_list_3[list_idx]
        list_idx    = list_idx + 1

    # remap AEv to log10
    AEv_0 = np.log10(AEv_0)
    AEv_1 = np.log10(AEv_1)
    AEv_2 = np.log10(AEv_2)
    AEv_3 = np.log10(AEv_3)


    # retrieve minimal and maximal values for plotting
    AE_max_0 = np.amax(AEv_0)
    AE_max_1 = np.amax(AEv_1)
    AE_max_2 = np.amax(AEv_2)
    AE_max_3 = np.amax(AEv_3)
    AE_min_0 = np.amin(AEv_0)
    AE_min_1 = np.amin(AEv_1)
    AE_min_2 = np.amin(AEv_2)
    AE_min_3 = np.amin(AEv_3)

    # print maximal values for comparison
    print('AE_max_0 = {}'.format(10**AE_max_0))
    print('AE_max_1 = {}'.format(10**AE_max_1))
    print('AE_max_2 = {}'.format(10**AE_max_2))
    print('AE_max_3 = {}'.format(10**AE_max_3))


    # find overall min and max
    AE_min = np.amin([AE_min_0,AE_min_1,AE_min_2,AE_min_3])
    AE_max = np.amax([AE_max_0,AE_max_1,AE_max_2,AE_max_3])

    lvls_res = 30

    levels0 = np.linspace(AE_min, AE_max, lvls_res)
    levels1 = np.linspace(AE_min, AE_max, lvls_res)
    levels2 = np.linspace(AE_min, AE_max, lvls_res)
    levels3 = np.linspace(AE_min, AE_max, lvls_res)
    fig, axs = plt.subplots(2,2, figsize=(6.850394, 5.0)) #figsize=(6.850394, 3.0)
    cnt0 = axs[0,0].contourf(kappav, deltav, AEv_0, levels=levels0, cmap='plasma',vmin=AE_min,vmax=AE_max)
    cnt1 = axs[0,1].contourf(kappav, deltav, AEv_1, levels=levels1, cmap='plasma',vmin=AE_min,vmax=AE_max)
    cnt2 = axs[1,0].contourf(kappav, deltav, AEv_2, levels=levels2, cmap='plasma',vmin=AE_min,vmax=AE_max)
    cnt3 = axs[1,1].contourf(kappav, deltav, AEv_3, levels=levels3, cmap='plasma',vmin=AE_min,vmax=AE_max)
    for c in cnt0.collections:
        c.set_edgecolor("face")
    for c in cnt1.collections:
        c.set_edgecolor("face")
    for c in cnt2.collections:
        c.set_edgecolor("face")
    for c in cnt3.collections:
        c.set_edgecolor("face")
    cbar0 = fig.colorbar(cnt0,ticks=[AE_min, AE_max],ax=axs.ravel().tolist())
    cbar0.set_ticklabels([fmt(AE_min,1), fmt(AE_max,1)])
    # cbar1 = fig.colorbar(cnt1,ticks=[AE_min, AE_max],ax=axs[0,1])
    # cbar1.set_ticklabels([fmt(AE_min,1), fmt(AE_max,1)])
    # cbar2 = fig.colorbar(cnt2,ticks=[AE_min, AE_max],ax=axs[1,0])
    # cbar2.set_ticklabels([fmt(AE_min,1), fmt(AE_max,1)])
    # cbar3 = fig.colorbar(cnt3,ticks=[AE_min, AE_max],ax=axs[1,1])
    # cbar3.set_ticklabels([fmt(AE_min,1), fmt(AE_max,1)])
    
    cbar0.set_label(r'$\log_{10} \widehat{A}$')
    # cbar1.set_label(r'$\log_{10} \widehat{A}$')
    # cbar3.set_label(r'$\log_{10} \widehat{A}$')
    # cbar0.solids.set_edgecolor("face")
    # cbar1.solids.set_edgecolor("face")
    # cbar2.solids.set_edgecolor("face")
    # cbar3.solids.set_edgecolor("face")
    # plt.title(r'Available Energy as a function of $s$ and $\alpha$' '\n' r'$\omega_n$={}, $\eta$={}, $\epsilon$={}, $q$={}, $d R_0/ d r$={}, $\kappa$={}, $s_\kappa$={}, $\delta$={}, $s_\delta$={}' '\n'.format(omn,eta,epsilon,q,dR0dr,kappa,s_kappa,delta,s_delta))
    axs[0,0].set_xlabel(r'$\kappa$')
    axs[0,1].set_xlabel(r'$\kappa$')
    axs[1,0].set_xlabel(r'$\kappa$')
    axs[1,1].set_xlabel(r'$\kappa$')
    axs[0,0].set_ylabel(r'$\delta$')
    axs[0,1].set_ylabel(r'$\delta$')
    axs[1,0].set_ylabel(r'$\delta$')
    axs[1,1].set_ylabel(r'$\delta$')
    axs[0,0].text(1.7, -0.6, r'$(a)$',c='white')
    axs[0,1].text(1.7, -0.6, r'$(b)$',c='white')
    axs[1,0].text(1.7, -0.6, r'$(c)$',c='white')
    axs[1,1].text(1.7, -0.6, r'$(d)$',c='white')
    axs[0,0].xaxis.set_tick_params(which='major', direction='in', top='on')
    axs[0,0].xaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[0,0].yaxis.set_tick_params(which='major', direction='in', top='on')
    axs[0,0].yaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[1,0].xaxis.set_tick_params(which='major', direction='in', top='on')
    axs[1,0].xaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[1,0].yaxis.set_tick_params(which='major', direction='in', top='on')
    axs[1,0].yaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[0,1].xaxis.set_tick_params(which='major', direction='in', top='on')
    axs[0,1].xaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[0,1].yaxis.set_tick_params(which='major', direction='in', top='on')
    axs[0,1].yaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[1,1].xaxis.set_tick_params(which='major', direction='in', top='on')
    axs[1,1].xaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[1,1].yaxis.set_tick_params(which='major', direction='in', top='on')
    axs[1,1].yaxis.set_tick_params(which='minor', direction='in', top='on')


    axs[0,0].set_xticklabels([])
    axs[0,1].set_xticklabels([])
    axs[1,1].set_yticklabels([])
    axs[0,1].set_yticklabels([])
    # plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/delta-kappa/delta-kappa_paper.png', format='png',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    plt.show()
