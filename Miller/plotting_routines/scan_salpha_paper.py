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
delta   = 0.5
dR0dr   = 0.0
s_q     = 'scan'
s_kappa = 0.0
s_delta = 0.0
alpha   = 'scan'
theta_res   = int(1e3+1)
L_ref       = 'major'




def fmt(x, pos=1):
    x = np.round(x,pos)
    return r'${}$'.format(x)


res = 100

# Construct grid for total integral
s_grid          =  np.linspace(-1.0, +5.0, num=res)
alpha_grid      =  np.linspace(+0.0, +2.0, num=res)


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
    AE_list_0 = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,  q, kappa, delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,L_ref) for idx, val in np.ndenumerate(AEv_0)])

    AE_list_1 = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,1.0, kappa, delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,L_ref) for idx, val in np.ndenumerate(AEv_1)])

    AE_list_2 = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,  q,   0.5, delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,L_ref) for idx, val in np.ndenumerate(AEv_2)])

    AE_list_3 = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,  q, kappa,-delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,L_ref) for idx, val in np.ndenumerate(AEv_3)])
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

    # remap AE to log10
    AEv_0 = np.nan_to_num(np.log10(AEv_0),neginf=-20)
    AEv_1 = np.nan_to_num(np.log10(AEv_1),neginf=-20)
    AEv_2 = np.nan_to_num(np.log10(AEv_2),neginf=-20)
    AEv_3 = np.nan_to_num(np.log10(AEv_3),neginf=-20)

    # find max and min values for colorbar
    AE_max_0 = np.amax(AEv_0)
    AE_max_1 = np.amax(AEv_1)
    AE_max_2 = np.amax(AEv_2)
    AE_max_3 = np.amax(AEv_3)
    AE_min_0 = np.amin(AEv_0)
    AE_min_1 = np.amin(AEv_1)
    AE_min_2 = np.amin(AEv_2)
    AE_min_3 = np.amin(AEv_3)

    # find overall max and min values for colorbar
    AE_max = np.max([AE_max_0,AE_max_1,AE_max_2,AE_max_3])
    # override min since log10(0) = -inf
    AE_min = -3 #np.min([AE_min_0,AE_min_1,AE_min_2,AE_min_3])

    # set contour levels
    levels0 = np.linspace(AE_min, AE_max, 30)
    levels1 = np.linspace(AE_min, AE_max, 30)
    levels2 = np.linspace(AE_min, AE_max, 30)
    levels3 = np.linspace(AE_min, AE_max, 30)

    # make figure
    fig, axs = plt.subplots(2,2, figsize=(6.850394, 5.0))

    # plot data
    cnt0 = axs[0,0].contourf(alphav, sv, AEv_0, levels=levels0, cmap='plasma',extend='min')
    cnt1 = axs[0,1].contourf(alphav, sv, AEv_1, levels=levels1, cmap='plasma',extend='min')
    cnt2 = axs[1,0].contourf(alphav, sv, AEv_2, levels=levels2, cmap='plasma')
    cnt3 = axs[1,1].contourf(alphav, sv, AEv_3, levels=levels3, cmap='plasma',extend='min')

    # set edgecolor to facecolor
    for c in cnt0.collections:
        c.set_edgecolor("face")
    for c in cnt1.collections:
        c.set_edgecolor("face")
    for c in cnt2.collections:
        c.set_edgecolor("face")
    for c in cnt3.collections:
        c.set_edgecolor("face")

    # set colorbar
    cbar0 = fig.colorbar(cnt0,ticks=[AE_min, AE_max],ax=axs.ravel().tolist())
    # cbar1 = fig.colorbar(cnt1,ticks=[AE_min, AE_max],ax=axs[0,1])
    # cbar2 = fig.colorbar(cnt2,ticks=[AE_min, AE_max],ax=axs[1,0])
    # cbar3 = fig.colorbar(cnt3,ticks=[AE_min, AE_max],ax=axs[1,1])

    # set colorbar ticks
    cbar0.set_ticklabels([fmt(AE_min,1), fmt(AE_max,1)])
    # cbar1.set_ticklabels([fmt(AE_min,1), fmt(AE_max,1)])
    # cbar2.set_ticklabels([fmt(AE_min,1), fmt(AE_max,1)])
    # cbar3.set_ticklabels([fmt(AE_min,1), fmt(AE_max,1)])

    # set label, only for rightmost colorbar
    cbar0.set_label(r'$\log_{10}\widehat{A}$')
    # cbar1.set_label(r'$\log_{10}\widehat{A}$')
    # cbar3.set_label(r'$\log_{10}\widehat{A}$')

    # set edgecolor to facecolor
    cbar0.solids.set_edgecolor("face")
    # cbar1.solids.set_edgecolor("face")
    # cbar2.solids.set_edgecolor("face")
    # cbar3.solids.set_edgecolor("face")
    
    # set labels
    axs[0,0].set_xlabel(r'$\alpha$')
    axs[1,0].set_xlabel(r'$\alpha$')
    axs[0,1].set_xlabel(r'$\alpha$')
    axs[1,1].set_xlabel(r'$\alpha$')
    axs[0,0].set_ylabel(r'$s$')
    axs[1,0].set_ylabel(r'$s$')
    axs[0,1].set_ylabel(r'$s$')
    axs[1,1].set_ylabel(r'$s$')

    # set plot names
    axs[0,0].text(1.7, -0.5, r'$(a)$',c='white')
    axs[0,1].text(1.7, -0.5, r'$(b)$',c='white')
    axs[1,0].text(1.7, -0.5, r'$(c)$',c='white')
    axs[1,1].text(1.7, -0.5, r'$(d)$',c='white')

    # set ticks
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

    # set ticklabels
    axs[0,0].set_xticklabels([])
    axs[0,1].set_xticklabels([])
    axs[1,1].set_yticklabels([])
    axs[0,1].set_yticklabels([])

    # save figure
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/s-alpha/s-alpha_paper.png', format='png',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    
    # show figure
    plt.show()
