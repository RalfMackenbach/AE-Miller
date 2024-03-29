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







omn = 3.0
q=2.0
L_ref = 'major'
epsilon = 1/3


res = 100

# Construct grid for total integral
s_grid      =  np.linspace(-5.0, +5.0,   num=res)
alpha_grid   = np.linspace(-5.0, +5.0,   num=res)


sv, alphav     = np.meshgrid(s_grid, alpha_grid, indexing='ij')
AEv0           = np.empty_like(sv)
AEv1           = np.empty_like(sv)
AEv_err        = np.empty_like(sv)






def fmt(x, pos=1):
    x = np.round(x,pos)
    return r'${}$'.format(x)

if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list0 = pool.starmap(AEtok.calc_AE_salpha, [(omn,0.0,epsilon,q,sv[idx],alphav[idx],L_ref) for idx, val in np.ndenumerate(sv)])
    AE_list1 = pool.starmap(AEtok.calc_AE_salpha, [(omn/1e10,1e10,epsilon,q,sv[idx],alphav[idx],L_ref) for idx, val in np.ndenumerate(sv)])
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

    AE0plot = np.log10(AEv0)
    AE1plot = np.log10(AEv1)

    max_level = np.max([AE0plot.max(),AE1plot.max()])
    min_level = np.min([AE0plot.min(),AE1plot.min()])

    levels0 = np.linspace(min_level, max_level, 30)
    levels1 = np.linspace(min_level, max_level, 30)

    fig, axs = plt.subplots(1,2, figsize=(6.850394, 5.0/2.1))
    fig.tight_layout()
    cnt0 = axs[0].contourf(alphav, sv, AE0plot, levels=levels0, cmap='plasma')
    cnt1 = axs[1].contourf(alphav, sv, AE1plot, levels=levels1, cmap='plasma')
    for c in cnt0.collections:
        c.set_edgecolor("face")
    for c in cnt1.collections:
        c.set_edgecolor("face")
    cbar0 = fig.colorbar(cnt0,ticks=[min_level,max_level],ax=axs[:])
    cbar0.set_ticklabels([fmt(min_level),fmt(max_level)])
    cbar0.solids.set_edgecolor("face")
    cbar0.set_label(r'$\log_{10}\widehat{A}$')
    axs[0].set_xlabel(r'$\alpha$')
    axs[1].set_xlabel(r'$\alpha$')
    axs[0].set_ylabel(r'$s$')
    axs[1].set_ylabel(r'$s$')

    axs[0].text(3.5, -3.5, r'$(a)$',c='white',horizontalalignment='center',verticalalignment='center')
    axs[1].text(3.5, -3.5, r'$(b)$',c='white',horizontalalignment='center',verticalalignment='center')

    # plt.text(3.2, -1.6, r'$(b)$',c='white')
    axs[0].xaxis.set_tick_params(which='major', direction='in', top='on')
    axs[0].xaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[0].yaxis.set_tick_params(which='major', direction='in', top='on')
    axs[0].yaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[1].xaxis.set_tick_params(which='major', direction='in', top='on')
    axs[1].xaxis.set_tick_params(which='minor', direction='in', top='on')
    axs[1].yaxis.set_tick_params(which='major', direction='in', top='on')
    axs[1].yaxis.set_tick_params(which='minor', direction='in', top='on')
    # plt.subplots_adjust(left=0.15, right=0.88, top=0.96, bottom=0.14)
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/s-alpha/s-alpha_paper.png', format='png',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    plt.show()
