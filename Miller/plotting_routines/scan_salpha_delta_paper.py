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
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

omn     = 1.0
eta     = 0.0
epsilon = 1/3
q       = 3.0
kappa   = 1.5
delta_bound = 0.2
dR0dr   =-0.5
s_q     = 0.0
s_kappa = 0.0
s_delta = 0.0
alpha   = 0.0
theta_res   = int(1e2)
lam_res     = int(1e2+1)
del_sign    = 0.0
L_ref       = 'major'




def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    if b != 0:
        return r'${} \cdot 10^{{{}}}$'.format(a, b)
    if b == 0:
        return r'${}$'.format(a)


res = 100
# Construct grid for total integral
s_grid          =  np.linspace(-1.0, +4.0, num=res, dtype='float64')
alpha_grid      =  np.linspace(+0.0, +0.5, num=res, dtype='float64')
omn_grid        =  np.logspace(-1.0, +1.0, num=res, dtype='float64')
q_grid        =  np.linspace(+1.0,   +4.0, num=res, dtype='float64')


sv, alphav = np.meshgrid(s_grid, alpha_grid, indexing='ij')
omnv, qv = np.meshgrid(omn_grid, q_grid, indexing='ij')
AEv_neg_0       = np.empty_like(sv)
AEv_neg_1       = np.empty_like(sv)
AEv_neg_2       = np.empty_like(sv)
AEv_neg_3       = np.empty_like(sv)
AEv_pos_0       = np.empty_like(sv)
AEv_pos_1       = np.empty_like(sv)
AEv_pos_2       = np.empty_like(sv)
AEv_pos_3       = np.empty_like(sv)



if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral

    for delta in [-delta_bound,delta_bound]:
        start_time = time.time()            #    omn,eta,epsilon,q,kappa,delta,dR0dr,s_q[idx],s_kappa[idx],s_delta[idx],alpha[idx],theta_res,lam_res,del_sign,L_ref
        AE_list0 = pool.starmap(AEtok.calc_AE, [(0.5,eta,1e-2, 1.0,1.5  ,delta, 0.0 ,sv[idx],    0.0,     0.1*delta/np.sqrt(1-delta**2),alphav[idx],theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(AEv_neg_0)])
        AE_list1 = pool.starmap(AEtok.calc_AE, [(2.0,eta,1/3 , 3.0,1.5  ,delta,-0.5 ,sv[idx],    0.5,     0.1*delta/np.sqrt(1-delta**2),alphav[idx],theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(AEv_neg_1)])
        AE_list2 = pool.starmap(AEtok.calc_AE, [(omnv[idx],eta,1e-2,qv[idx],1.5,delta, 0.0,0.0,0.0,0.1*delta/np.sqrt(1-delta**2),0.0,theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(AEv_neg_2)])
        AE_list3 = pool.starmap(AEtok.calc_AE, [(omnv[idx],eta,1/3 ,qv[idx],1.5,delta,-0.5,3.0,0.5,0.1*delta/np.sqrt(1-delta**2),0.5,theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(AEv_neg_3)])
        print("data generated in       --- %s seconds ---" % (time.time() - start_time))

        # reorder data full int
        list_idx = 0

        if delta == -delta_bound:
            for idx, val in np.ndenumerate(AEv_neg_0):
                AEv_neg_0[idx]    = AE_list0[list_idx]
                AEv_neg_1[idx]    = AE_list1[list_idx]
                AEv_neg_2[idx]    = AE_list2[list_idx]
                AEv_neg_3[idx]    = AE_list3[list_idx]
                list_idx    = list_idx + 1

        if delta == delta_bound:
            for idx, val in np.ndenumerate(AEv_pos_0):
                AEv_pos_0[idx]    = AE_list0[list_idx]
                AEv_pos_1[idx]    = AE_list1[list_idx]
                AEv_pos_2[idx]    = AE_list2[list_idx]
                AEv_pos_3[idx]    = AE_list3[list_idx]
                list_idx    = list_idx + 1

    pool.close()

    delta_0 = np.log(AEv_neg_0/AEv_pos_0)
    delta_1 = np.log(AEv_neg_1/AEv_pos_1)
    delta_2 = np.log(AEv_neg_2/AEv_pos_2)
    delta_3 = np.log(AEv_neg_3/AEv_pos_3)

    fig, axs = plt.subplots(2,2, figsize=(6.850394, 5.0)) #figsize=(6.850394, 3.0)
    max_0 = np.nanmax([np.nanmax(delta_0),-np.nanmin(delta_0)])
    max_1 = np.nanmax([np.nanmax(delta_1),-np.nanmin(delta_1)])
    max_2 = np.nanmax([np.nanmax(delta_2),-np.nanmin(delta_2)])
    max_3 = np.nanmax([np.nanmax(delta_3),-np.nanmin(delta_3)])

    n_lvls = 30
    levels0 = np.linspace(-max_0,max_0,n_lvls)
    levels1 = np.linspace(-max_1,max_1,n_lvls)
    levels2 = np.linspace(-max_2,max_2,n_lvls)
    levels3 = np.linspace(-max_3,max_3,n_lvls)


    cnt0 = axs[0,0].contourf(alphav, sv, delta_0, levels=levels0, cmap='bwr')
    cnl0 = axs[0,0].contour (alphav, sv, delta_0, [0.0],cmap='gray')
    cnt1 = axs[0,1].contourf(alphav, sv, delta_1, levels=levels1, cmap='bwr')
    cnl1 = axs[0,1].contour (alphav, sv, delta_1, [0.0],cmap='gray')
    cnt2 = axs[1,0].contourf(qv, omnv, delta_2, levels=levels2, cmap='bwr')
    cnl2 = axs[1,0].contour (qv, omnv, delta_2, [0.0],cmap='gray')
    cnt3 = axs[1,1].contourf(qv, omnv, delta_3, levels=levels3, cmap='bwr')
    cnl3 = axs[1,1].contour (qv, omnv, delta_3, [0.0],cmap='gray')
    axs[1,1].set_yscale('log')
    axs[1,0].set_yscale('log')

    axs[0,0].clabel(cnl0)
    axs[0,1].clabel(cnl1)
    axs[1,0].clabel(cnl2)
    axs[1,1].clabel(cnl3)


    for c in cnt0.collections:
        c.set_edgecolor("face")
    for c in cnt1.collections:
        c.set_edgecolor("face")
    for c in cnt2.collections:
        c.set_edgecolor("face")
    for c in cnt3.collections:
        c.set_edgecolor("face")
    cbar0 = fig.colorbar(cnt0,ticks=[-max_0,max_0],ax=axs[0,0])
    cbar0.set_ticklabels([r'${}$'.format(np.round(-max_0,1)),r'${}$'.format(np.round(max_0,1))])
    cbar1 = fig.colorbar(cnt1,ticks=[-max_1,max_1],ax=axs[0,1])
    cbar1.set_ticklabels([r'${}$'.format(np.round(-max_1,1)),r'${}$'.format(np.round(max_1,1))])
    cbar2 = fig.colorbar(cnt2,ticks=[-max_2,max_2],ax=axs[1,0])
    cbar2.set_ticklabels([r'${}$'.format(np.round(-max_2,1)),r'${}$'.format(np.round(max_2,1))])
    cbar3 = fig.colorbar(cnt3,ticks=[-max_3,max_3],ax=axs[1,1])
    cbar3.set_ticklabels([r'${}$'.format(np.round(-max_3,1)),r'${}$'.format(np.round(max_3,1))])

    cbar1.set_label(r'$\log(\Delta)$')
    cbar3.set_label(r'$\log(\Delta)$')
    # cbar.set_label(r'$\widehat{A}$')
    cbar0.solids.set_edgecolor("face")
    cbar1.solids.set_edgecolor("face")
    cbar2.solids.set_edgecolor("face")
    cbar3.solids.set_edgecolor("face")
    # plt.title(r'Available Energy as a function of $s$ and $\alpha$' '\n' r'$\omega_n$={}, $\eta$={}, $\epsilon$={}, $q$={}, $d R_0/ d r$={}, $\kappa$={}, $s_\kappa$={}, $\delta$={}, $s_\delta$={}' '\n'.format(omn,eta,epsilon,q,dR0dr,kappa,s_kappa,delta,s_delta))
    axs[0,1].set_xlabel(r'$\alpha$')
    axs[0,0].set_xlabel(r'$\alpha$')
    axs[1,0].set_xlabel(r'$q$')
    axs[1,1].set_xlabel(r'$q$')
    axs[0,0].set_ylabel(r'$s$')
    axs[1,0].set_ylabel(r'$\hat{\omega}_n$')
    axs[0,0].text(0.05, 3.4, r'$(a)$',c='black',va='center',ha='center')
    axs[0,1].text(0.05, 3.4, r'$(b)$',c='black',va='center',ha='center')
    axs[1,0].text(1.3,  6.0, r'$(c)$',c='black',va='center',ha='center')
    axs[1,1].text(1.3,  6.0, r'$(d)$',c='black',va='center',ha='center')
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
    axs[0,1].set_yticklabels([])
    axs[1,1].set_yticklabels([])
    axs[0,0].set_title(r'$\mathrm{core}$')
    axs[0,1].set_title(r'$\mathrm{edge}$')
    plt.tight_layout()
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/s-alpha/delta_paper.eps', format='eps',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    plt.show()
