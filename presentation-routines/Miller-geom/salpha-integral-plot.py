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


log_scale = True


omn = 3.0
eta= 0.0
epsilon = 1/3
q  = 2.0
kappa = 1.0
delta = 0.0
drR0 = 0.0
s_kappa = 0.0
s_delta = 0.0
theta_res = 1002




res = 30


# Construct grid for total integral
s_grid      =  np.linspace(-5.0, +5.0,   num=res)
alpha_grid   = np.linspace(-5.0, +5.0,   num=res)


sv, alphav     = np.meshgrid(s_grid, alpha_grid, indexing='ij')
AEv            = np.empty_like(sv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))


    # time the full integral
    start_time = time.time()
    AE_list = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,q,kappa,delta,drR0,sv[idx],s_kappa,s_delta,alphav[idx],theta_res) for idx, val in np.ndenumerate(sv)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list = np.asarray(AE_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(sv):
        AEv[idx]    = AE_list[list_idx]
        list_idx = list_idx + 1

    if log_scale:
        AEv = np.log10(AEv)

    scale=0.7
    fig, ax = plt.subplots(1,1, figsize=(scale*6.850394, scale*5.0))
    if log_scale:
        AE_min = np.min(AEv)
        AE_max = np.max(AEv)
        extender = None
        # if AE_min is -inf, set it to one standard deviation below the mean in non-log space
        if AE_min == -np.inf:
            AE_min_nonlog = np.mean(10**AEv) - np.std(10**AEv)
            AE_min = np.log10(AE_min_nonlog)
            print('AE_min is -inf, setting to {}'.format(AE_min))
            extender = 'min'
            # set all -inf values to AE_min - 1
            AEv[AEv == -np.inf] = AE_min - 1
    if not log_scale:
        AE_min = 0.0
        AE_max = np.max(AEv)
        extender = None

    levels = np.linspace(AE_min, AE_max, 25)

    cnt = plt.contourf(alphav, sv, AEv, levels=levels, cmap='plasma', vmin=AE_min, vmax=AE_max,extend=extender)
    for c in cnt.collections:
        c.set_edgecolor("face")
    cbar = plt.colorbar()
    if log_scale:
        cbar.set_label(r'$\log_{10}\widehat{A}$')
    if not log_scale:
        cbar.set_label(r'$\widehat{A}$')

    cbar.solids.set_edgecolor("face")
    plt.xlabel(r'pressure gradient, $\alpha$')
    plt.ylabel(r'magnetic shear, $s$')
    ax.xaxis.set_tick_params(which='major', direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', direction='in', top='on')
    ax.yaxis.set_tick_params(which='minor', direction='in', top='on')
    plt.tight_layout()
    plt.show()
