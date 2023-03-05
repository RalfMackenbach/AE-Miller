import AEtok.AE_tokamak_calculation as AEtok
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import matplotlib        as mpl
from matplotlib import rc




rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.cbook as cbook


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)




# Construct grid for total integral
s_grid      =  np.linspace(-2.0, +2.0,   num=100)
alpha_grid   = np.linspace(+0.0, +2.0,   num=100)


sv, alphav     = np.meshgrid(s_grid, alpha_grid, indexing='ij')
AEv0            = np.empty_like(sv)
AEv1            = np.empty_like(sv)
AEv_err        = np.empty_like(sv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    omn = 3.0
    epsilon = 1e-6
    q_val = 2.0
    eta = 1.0
    L_ref = 'major'
    A = 3.0
    rho = 1e-6
    theta_res=int(1e3)
    lam_res_mill = int(1e3)
    lam_res_salp = int(1e5)

    # time the full integral
    start_time = time.time()
    AE_list0 = pool.starmap(AEtok.calc_AE_salpha, [(omn,eta,epsilon,q_val,sv[idx],alphav[idx],lam_res_salp,L_ref,A,rho) for idx, val in np.ndenumerate(sv)])
    print('s-alpha done')
    AE_list1 = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,q_val,1.0,0.0,0.0,sv[idx],0.0,0.0,alphav[idx],theta_res,lam_res_mill,L_ref,A,rho) for idx, val in np.ndenumerate(sv)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list0 = np.asarray(AE_list0)
    AE_list1 = np.asarray(AE_list1)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(sv):
        AEv0[idx]    = AE_list0[list_idx]
        AEv1[idx]    = AE_list1[list_idx]
        list_idx = list_idx + 1

    rel_err = np.abs((AEv1-AEv0)/AEv0)
    print('max AE s-alpha val is', np.amax(AEv0))
    print('max AE Miller val is', np.amax(AEv1))
    print('max err is', np.amax(rel_err))
    print('min err is', np.amin(rel_err))
    print('mean err is',np.mean(rel_err))

    levels0 = np.linspace(0, np.amax(AEv0), 25)
    levels1 = np.linspace(0, np.amax(AEv1), 25)

    fig, axs = plt.subplots(1,3, figsize=(6.850394, 5.0/2)) #figsize=(6.850394, 3.0)
    cnt0 = axs[0].contourf(alphav, sv, AEv0, levels=levels0, cmap='plasma')
    cnt1 = axs[1].contourf(alphav, sv, AEv1, levels=levels1, cmap='plasma')
    for c in cnt0.collections:
        c.set_edgecolor("face")
    for c in cnt1.collections:
        c.set_edgecolor("face")
    vmin_err = np.floor(np.log10(np.amin(rel_err)))
    vmax_err = np.floor(np.log10(np.amax(rel_err)))
    pcm = axs[2].pcolor(alphav, sv, rel_err,
                   norm=colors.LogNorm(vmin=10**vmin_err, vmax=10**vmax_err),
                   cmap='Greys', shading='auto')
    pcm.set_edgecolor('face')
    cbar0 = fig.colorbar(cnt0,ticks=[0.0, np.amax(AEv0)],ax=axs[0],orientation="horizontal",pad=0.2)
    cbar1 = fig.colorbar(cnt1,ticks=[0.0, np.amax(AEv1)],ax=axs[1],orientation="horizontal",pad=0.2)
    cbar2 = fig.colorbar(pcm, ticks=[10**vmin_err,10**vmax_err], ax=axs[2],orientation="horizontal",pad=0.2)
    cbar0.ax.set_xticklabels([r'$0$',fmt(np.amax(AEv0),1)])
    cbar1.ax.set_xticklabels([r'$0$',fmt(np.amax(AEv1),1)])
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
    axs[1].set_ylabel(r'$s$')
    axs[2].set_ylabel(r'$s$')


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
