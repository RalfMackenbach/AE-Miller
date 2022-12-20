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
q       = 1.0
kappa   = 1.5
delta_bound = 0.2
dR0dr   = 0.0
s_q     = 'scan'
s_kappa = 0.0
s_delta = 0.0
alpha   = 'scan'
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



# Construct grid for total integral
s_grid          =  np.linspace(-1.0, +4.0, num=50, dtype='float64')
alpha_grid      =  np.linspace(+0.0, +0.5, num=50, dtype='float64')


sv, alphav = np.meshgrid(s_grid, alpha_grid, indexing='ij')
AEv_neg        = np.empty_like(sv)
AEv_pos        = np.empty_like(sv)



if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral

    for delta in [-delta_bound,delta_bound]:
        start_time = time.time()
        AE_list = pool.starmap(AEtok.calc_AE, [(omn,eta,epsilon,q,kappa,delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(AEv_neg)])
        print("data generated in       --- %s seconds ---" % (time.time() - start_time))


        AE_list = np.asarray(AE_list)

        # reorder data full int
        list_idx = 0

        if delta == -delta_bound:
            for idx, val in np.ndenumerate(AEv_neg):
                AEv_neg[idx]    = AE_list[list_idx]
                list_idx    = list_idx + 1

        if delta == delta_bound:
            for idx, val in np.ndenumerate(AEv_pos):
                AEv_pos[idx]    = AE_list[list_idx]
                list_idx    = list_idx + 1

    pool.close()
    plot_arr = np.log(AEv_neg/AEv_pos)
    # levels = np.linspace(0, 4.0, 25)
    print('Max val is: ',np.nanmax(np.ma.masked_invalid(plot_arr)))
    print('Min val is: ',np.nanmin(plot_arr))
    minlog = np.nanmin(plot_arr)
    maxlog = np.nanmax(np.ma.masked_invalid(plot_arr))
    fig = plt.figure(figsize=(3.375, 2.3)) #figsize=(6.850394, 3.0)
    offset = mcolors.TwoSlopeNorm(vmin=minlog,vcenter=0.0, vmax=maxlog)
    res = 10
    levels1 = np.linspace(minlog,0.0,res,endpoint=False)
    levels2 = np.linspace(0.0,maxlog,res)
    levels = np.append(levels1,levels2)
    ax  = fig.gca()
    cnt = plt.contourf(alphav, sv, plot_arr, levels=levels, cmap='RdBu_r',norm = offset)
    for c in cnt.collections:
        c.set_edgecolor("face")
    cbar = plt.colorbar(ticks=[levels[0], 0.0, levels[-1]],norm=LogNorm())
    cbar.set_ticklabels([r'${}$'.format(np.round(levels[0],1)),r'$0$',r'${}$'.format(np.round(levels[-1],1))])
    cbar.set_label(r'$ \log \left( \Delta_\delta \right)$')
    # cbar.set_label(r'$\widehat{A}$')
    cbar.solids.set_edgecolor("face")
    # plt.title(r'Available Energy as a function of $s$ and $\alpha$' '\n' r'$\omega_n$={}, $\eta$={}, $\epsilon$={}, $q$={}, $d R_0/ d r$={}, $\kappa$={}, $s_\kappa$={}, $\delta$={}, $s_\delta$={}' '\n'.format(omn,eta,epsilon,q,dR0dr,kappa,s_kappa,delta,s_delta))
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$s$')
    cnl = plt.contour(alphav, sv, plot_arr, [0.0],cmap='gray')
    plt.clabel(cnl)
    # plt.text(3.2/4, -1.6, r'$(a)$',c='white')
    ax.xaxis.set_tick_params(which='major', direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', direction='in', top='on')
    ax.yaxis.set_tick_params(which='minor', direction='in', top='on')
    plt.tight_layout()
    # plt.subplots_adjust(left=0.15, right=0.88, top=0.96, bottom=0.14)
    plt.margins(0.1)
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/s-alpha/s-alpha-ratio.eps'.format(omn,eta,epsilon,q,kappa,delta,dR0dr,s_kappa,s_delta), format='eps',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    plt.show()
