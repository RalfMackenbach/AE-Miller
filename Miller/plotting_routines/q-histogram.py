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
import matplotlib.ticker as ticker
import random
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

random.seed(10)
np.random.seed(0)



def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)



# Construct grid for total integral

n_points = 1000

q_big = 1e4
q_sml = 1.0

omn = np.random.uniform(low=0.0, high=5.0, size=(n_points,))
eta = np.random.uniform(low=0.0, high=0.0, size=(n_points,))
epsilon = np.random.uniform(low=0.1, high=0.5, size=(n_points,))
kappa = np.random.uniform(low=1.0, high=2.0, size=(n_points,))
delta = np.random.uniform(low=-0.5, high=0.5, size=(n_points,))
s_q   = np.random.uniform(low=-1.0, high=3.0, size=(n_points,))
s_kappa = np.random.uniform(low=0.0, high=0.0, size=(n_points,))
s_delta = np.random.uniform(low=0.0, high=0.0, size=(n_points,))
dR0dr = np.random.uniform(low=0.0, high=0.0, size=(n_points,))
alpha = np.random.uniform(low=0.0, high=0.0, size=(n_points,))
theta_res   = int(1e2 +1)
lam_res     = int(1e3)
del_sign    = 0.0
L_ref       = 'minor'


AEv_big        = np.empty_like(omn)
AEv_sml        = np.empty_like(omn)



if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))
        

    # time the full integral
    start_time = time.time()
    AE_list_big = pool.starmap(AEtok.calc_AE, [(omn[idx],eta[idx],epsilon[idx],q_big,kappa[idx],delta[idx],dR0dr[idx],s_q[idx],s_kappa[idx],s_delta[idx],alpha[idx],theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(omn)])
    AE_list_sml = pool.starmap(AEtok.calc_AE, [(omn[idx],eta[idx],epsilon[idx],q_sml,kappa[idx],delta[idx],dR0dr[idx],s_q[idx],s_kappa[idx],s_delta[idx],alpha[idx],theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(omn)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))


    AE_list_big = np.asarray(AE_list_big)/q_big**2.0
    AE_list_sml = np.asarray(AE_list_sml)/q_sml**2.0

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv_big):
        AEv_big[idx]    = AE_list_big[list_idx]
        AEv_sml[idx]    = AE_list_sml[list_idx]
        list_idx    = list_idx + 1

    plot_arr = AEv_sml/AEv_big

    print('average value found is:', np.mean(plot_arr))

    fig, ax = plt.subplots(1,2, figsize=(6.850394, 2.3), facecolor='w')
    bins_arr = np.linspace(0.0,2.0,101)
    if np.amax(plot_arr)>2:
        bins_arr=np.append(bins_arr,np.amax(plot_arr))
    cnts, values, bars = ax[1].hist(plot_arr, bins=bins_arr,density=True,cumulative=True,lw=0)

    for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
        if value < 1:
            bar.set_facecolor('g')
            bar.set_hatch('x')
        if value >= 1:
            bar.set_facecolor('r')
            bar.set_hatch('/')
    ax[1].set_xlabel(r'$\Delta_q$')
    ax[1].set_ylabel(r'$\mathrm{CDF}$')
    ax[1].set_xlim(0.0,2.0)
    ax[1].set_ylim(0.0,1.0)
    ax[1].text(2*1/10, 1*7/8,r'$(b)$',ha='center',va='center')

    large_val = 0
    lines = ["-","--","-.",":"]
    omn = [1.0,1.0,1.0,1.0]
    eta = [0.0,0.0,0.0,0.0]
    epsilon = [0.1,0.2,0.3,0.5]
    kappa = [1.0,2.0,1.0,2.0]
    delta = [-0.2,0.2,0.0,0.5]
    s_q   = [-0.5,0.0,0.5,1.0]
    s_kappa = [0.0,0.0,0.0,0.0]
    s_delta = [0.0,0.0,0.0,0.0]
    dR0dr = [0.0,0.0,0.0,0.0]
    alpha = [0.0,0.0,0.0,0.0]
    theta_res   = int(1e2 +1)
    lam_res     = int(1e4)
    del_sign    = 0.0
    L_ref       = 'minor'
    for q_idx in [0,1,2,3]:
        start_time = time.time()
        q_arr = np.linspace(1e-3,3.0,100)
        AE_q = pool.starmap(AEtok.calc_AE, [(omn[q_idx],eta[q_idx],epsilon[q_idx],q_arr[idx],kappa[q_idx],delta[q_idx],dR0dr[q_idx],s_q[q_idx],s_kappa[q_idx],s_delta[q_idx],alpha[q_idx],theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(q_arr)])
        ax[0].plot(q_arr,AE_q/q_arr**2.0,lines[q_idx])
        max = np.amax(AE_q/q_arr**2.0)
        large_val = np.amax([large_val,max])
        print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    ax[0].set_xlim(0.0,3.0)
    ax[0].set_ylim(0.0,1.1*large_val)
    ax[0].text(3*1/10, 1.1*large_val*7/8 ,r'$(a)$',ha='center',va='center')
    ax[0].set_xlabel(r'$q$')
    ax[0].set_ylabel(r'$\widehat{A}/q^2$')

    ax[0].xaxis.set_tick_params(which='major', direction='in', top='on')
    ax[0].xaxis.set_tick_params(which='minor', direction='in', top='on')
    ax[0].yaxis.set_tick_params(which='major', direction='in', top='on')
    ax[0].yaxis.set_tick_params(which='minor', direction='in', top='on')
    ax[1].xaxis.set_tick_params(which='major', direction='in', top='on')
    ax[1].xaxis.set_tick_params(which='minor', direction='in', top='on')
    ax[1].yaxis.set_tick_params(which='major', direction='in', top='on')
    ax[1].yaxis.set_tick_params(which='minor', direction='in', top='on')

    plt.tight_layout()
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/q-dependence/q_distribution.eps', format='eps',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    
    

    pool.close()
    plt.show()

