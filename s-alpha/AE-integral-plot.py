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
q  = 2.0
eta= 0.0
L_ref = 'major'
epsilon = 1/3
A = 3.0
rho = 0.5





# Construct grid for total integral
s_grid      =  np.linspace(-5.0, +5.0,   num=200)
alpha_grid   = np.linspace(-5.0, +5.0,   num=200)


sv, alphav     = np.meshgrid(s_grid, alpha_grid, indexing='ij')
AEv            = np.empty_like(sv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list = pool.starmap(AEtok.calc_AE_salpha, [(omn,eta,epsilon,q,sv[idx],alphav[idx],L_ref,A,rho) for idx, val in np.ndenumerate(sv)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list = np.asarray(AE_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(sv):
        AEv[idx]    = AE_list[list_idx]
        list_idx = list_idx + 1


    scale=0.7
    fig, ax = plt.subplots(1,1, figsize=(scale*6.850394, scale*5.0))
    cnt = plt.contourf(alphav, sv, np.log(AEv), 25, cmap='plasma')
    for c in cnt.collections:
        c.set_edgecolor("face")
    cbar = plt.colorbar()
    cbar.set_label(r'$\ln \widehat{A}$')
    cbar.solids.set_edgecolor("face")
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$s$')
    ax.xaxis.set_tick_params(which='major', direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', direction='in', top='on')
    ax.yaxis.set_tick_params(which='minor', direction='in', top='on')
    plt.tight_layout()
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/s-alpha/s-alpha_paper.eps', format='eps',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight', pad_inches = 0.01)
    plt.show()
