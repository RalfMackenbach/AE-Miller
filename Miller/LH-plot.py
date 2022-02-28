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

omn     = 'scan'
eta     = 0.0
epsilon = 0.3
q       = 2.5
kappa   = 1.7
delta   = 0.7
dR0dr   = 0.0
s_kappa = 0.0
s_delta = 0.0
theta_res   = int(1e3 +1)
lam_res     = int(1e3)
del_sign    = 0.0


c_s     = 0.2
c_a     = 0.03




def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)



# Construct grid for total integral
omn_grid          =  np.linspace(0.0, +10.0, num=1000, dtype='float64')


omnv           = omn_grid
AEv            = np.empty_like(omnv)
AEv_err        = np.empty_like(omnv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list = pool.starmap(AEtok.calc_AE, [(omnv[idx],eta,epsilon,q,kappa,delta,dR0dr,2 - 2*omnv[idx]/10 ,s_kappa,s_delta, omnv[idx]/10,theta_res,lam_res,del_sign) for idx, val in np.ndenumerate(AEv)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list = np.asarray(AE_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv):
        AEv[idx]    = AE_list[list_idx]
        list_idx    = list_idx + 1

    theta = np.linspace(0,np.pi*2,theta_res)
    f = plt.figure(1)
    plt.plot(1 + epsilon*np.cos(theta + np.arcsin(delta)*np.sin(theta)), epsilon*kappa*np.sin(theta))
    plt.axis('equal')
    plt.title("Flux surface")
    plt.xlabel(r'$R/R_0$')
    plt.ylabel(r'$Z/R_0$')


    levels = np.linspace(0, 30, 25)
    # print(np.amax(AEv))
    fig = plt.figure(figsize=(6, 2.3)) #figsize=(3.375, 2.3)
    ax  = fig.gca()
    cnt = plt.plot(omnv, AEv,'black')
    # plt.title(r'Available Energy as a function of $s$ and $\alpha$' '\n' r'$\omega_n$={}, $\eta$={}, $\epsilon$={}, $q$={}, $d R_0/ d r$={}, $\kappa$={}, $s_\kappa$={}, $\delta$={}, $s_\delta$={}' '\n'.format(omn,eta,epsilon,q,dR0dr,kappa,s_kappa,delta,s_delta))
    plt.xlabel(r'$\hat{\omega}_n$')
    plt.ylabel(r'$\widehat{A}$')
    # plt.text(3.2, -1.6, r'$(b)$',c='white')
    ax.xaxis.set_tick_params(which='major', direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', direction='in', top='on')
    ax.yaxis.set_tick_params(which='minor', direction='in', top='on')
    ax.set_xlim(0,10)
    ax.set_ylim(0,8)
    ax.grid(True)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.15, right=0.88, top=0.96, bottom=0.14)
    plt.margins(0.1)
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/LH_plot.eps', format='eps',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight', pad_inches = 0.1)
    plt.show()
