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
kappa   = 1.7
delta   = 0.5
dR0dr   = 0.0
s_q     = 'scan'
s_kappa = 0.0
s_delta = 0.0
alpha   = 'scan'
theta_res   = int(1e2 +1)
lam_res     = int(1e3)
del_sign    = 0.0




def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)



# Construct grid for total integral
s_grid          =  np.linspace(-2.0, +2.0, num=100, dtype='float64')
alpha_grid      =  np.linspace(+0.0, +2.0, num=100, dtype='float64')


sv, alphav = np.meshgrid(s_grid, alpha_grid, indexing='ij')
AEv_0   = np.empty_like(sv)
AEv_1   = np.empty_like(sv)
AEv_2   = np.empty_like(sv)
AEv_3   = np.empty_like(sv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))


    # time the full integral
    start_time = time.time()
    AE_list_0 = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,  q, kappa, delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,lam_res,del_sign) for idx, val in np.ndenumerate(AEv_0)])

    AE_list_1 = pool.starmap(AEtok.calc_AE, [(omn,eta,    0.7,  q, kappa, delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,lam_res,del_sign) for idx, val in np.ndenumerate(AEv_1)])

    AE_list_2 = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,1.0, kappa, delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,lam_res,del_sign) for idx, val in np.ndenumerate(AEv_2)])

    AE_list_3 = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,  q, kappa,-delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,lam_res,del_sign) for idx, val in np.ndenumerate(AEv_3)])
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




    # theta = np.linspace(0,np.pi*2,theta_res)
    # f = plt.figure(1)
    # plt.plot(1 + epsilon*np.cos(theta + np.arcsin(delta)*np.sin(theta)), epsilon*kappa*np.sin(theta))
    # plt.axis('equal')
    # plt.title("Flux surface")
    # plt.xlabel(r'$R/R_0$')
    # plt.ylabel(r'$Z/R_0$')

    # print(np.amax(AEv))
    AE_max_0 = np.amax(AEv_0)
    AE_max_1 = np.amax(AEv_1)
    AE_max_2 = np.amax(AEv_2)
    AE_max_3 = np.amax(AEv_3)
    levels0 = np.linspace(0, AE_max_0**(3/2), 25)
    levels1 = np.linspace(0, AE_max_1**(3/2), 25)
    levels2 = np.linspace(0, AE_max_2**(3/2), 25)
    levels3 = np.linspace(0, AE_max_3**(3/2), 25)
    fig, axs = plt.subplots(2,2, figsize=(6.850394, 5.0)) #figsize=(6.850394, 3.0)
    cnt0 = axs[0,0].contourf(alphav, sv, AEv_0**(3/2), levels=levels0, cmap='plasma')
    cnt1 = axs[0,1].contourf(alphav, sv, AEv_1**(3/2), levels=levels1, cmap='plasma')
    cnt2 = axs[1,0].contourf(alphav, sv, AEv_2**(3/2), levels=levels2, cmap='plasma')
    cnt3 = axs[1,1].contourf(alphav, sv, AEv_3**(3/2), levels=levels3, cmap='plasma')
    for c in cnt0.collections:
        c.set_edgecolor("face")
    for c in cnt1.collections:
        c.set_edgecolor("face")
    for c in cnt2.collections:
        c.set_edgecolor("face")
    for c in cnt3.collections:
        c.set_edgecolor("face")
    cbar0 = fig.colorbar(cnt0,ticks=[0.0, AE_max_0**(3/2)],ax=axs[0,0])
    cbar1 = fig.colorbar(cnt1,ticks=[0.0, AE_max_1**(3/2)],ax=axs[0,1])
    cbar2 = fig.colorbar(cnt2,ticks=[0.0, AE_max_2**(3/2)],ax=axs[1,0])
    cbar3 = fig.colorbar(cnt3,ticks=[0.0, AE_max_3**(3/2)],ax=axs[1,1])

    cbar1.set_label(r'$\widehat{A}^{3/2}$')
    cbar3.set_label(r'$\widehat{A}^{3/2}$')
    # cbar.set_label(r'$\widehat{A}$')
    cbar0.solids.set_edgecolor("face")
    cbar1.solids.set_edgecolor("face")
    cbar2.solids.set_edgecolor("face")
    cbar3.solids.set_edgecolor("face")
    # plt.title(r'Available Energy as a function of $s$ and $\alpha$' '\n' r'$\omega_n$={}, $\eta$={}, $\epsilon$={}, $q$={}, $d R_0/ d r$={}, $\kappa$={}, $s_\kappa$={}, $\delta$={}, $s_\delta$={}' '\n'.format(omn,eta,epsilon,q,dR0dr,kappa,s_kappa,delta,s_delta))
    axs[1,0].set_xlabel(r'$\alpha$')
    axs[1,1].set_xlabel(r'$\alpha$')
    axs[0,0].set_ylabel(r'$s$')
    axs[1,0].set_ylabel(r'$s$')
    axs[0,0].text(1.7, -1.5, r'$(a)$',c='white')
    axs[0,1].text(1.7, -1.5, r'$(b)$',c='white')
    axs[1,0].text(1.7, -1.5, r'$(c)$',c='white')
    axs[1,1].text(1.7, -1.5, r'$(d)$',c='white')
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
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/s-alpha/s-alpha_paper.eps', format='eps',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    plt.show()
