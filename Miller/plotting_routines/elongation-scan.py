import AEtok.AE_tokamak_calculation as AEtok
import numpy                        as np
import multiprocessing              as mp
import time
import matplotlib.pyplot            as plt
import h5py
import matplotlib        as mpl
from matplotlib import rc
import matplotlib.ticker as ticker
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

omn     = 1.0
eta     = 0.0
epsilon = 1e-5
q       = 3.0
kappa   = np.logspace(-2,2,16)
s_q     = 0.0
delta   = 0.0
dR0dr   = 0.0
s_kappa = 0.0
s_delta = 0.0
alpha   = 0.0
theta_res   = int(1e4+1)
L_ref       = 'major'



def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)




AEv_1          = np.empty_like(kappa)
AEv_2          = np.empty_like(kappa)
AEv_err        = np.empty_like(kappa)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list_1 = pool.starmap(AEtok.calc_AE, [(omn/1e2  * np.sqrt(val) ,eta,epsilon/np.sqrt(val),q,val,delta,dR0dr,s_q ,s_kappa,s_delta, alpha,theta_res,L_ref) for idx, val in np.ndenumerate(kappa)])
    AE_list_2 = pool.starmap(AEtok.calc_AE, [(omn*1e2 * np.sqrt(val) ,eta,epsilon/np.sqrt(val),q,val,delta,dR0dr,s_q ,s_kappa,s_delta, alpha,theta_res,L_ref) for idx, val in np.ndenumerate(kappa)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list_1 = np.asarray(AE_list_1)
    AE_list_2 = np.asarray(AE_list_2)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv_err):
        AEv_1[idx]      = AE_list_1[list_idx] 
        AEv_2[idx]      = AE_list_2[list_idx] 
        list_idx        = list_idx + 1


    # make two plots side by side, set constrained layout
    fig, axs = plt.subplots(1,2, figsize=(6.850394,2.5), constrained_layout=True)

    # plot the first one
    axs[0].loglog(kappa, AEv_1)
    axs[1].loglog(kappa, AEv_2)

    # set title to "weakly driven" and "strongly driven"
    axs[0].set_title('Weakly driven')
    axs[1].set_title('Strongly driven')


    # add two power laws, kappa^(-1/4) and kappa^(+3/4)
    # for kappa^(-1/4) we want it to line up with the first point
    # for kappa^(+3/4) we want it to line up with the last point
    # so we need to find the prefactors
    prefactor_1 = AEv_1[+0] * kappa[+0]**(-1/4)
    prefactor_2 = AEv_2[-1] * kappa[-1]**(+3/4)
    # dashed and dotted lines
    axs[0].loglog(kappa, prefactor_1 * kappa**(+1/4), '--', color='black', label=r'$\kappa^{+1/4}$')
    axs[1].loglog(kappa, prefactor_2 * kappa**(-3/4), ':', color='black', label=r'$\kappa^{-3/4}$')

    # set axes labels
    axs[0].set_xlabel(r'$\kappa$')
    axs[1].set_xlabel(r'$\kappa$')
    axs[0].set_ylabel(r'$\widehat{A}$')

    # add legend
    axs[0].legend()
    axs[1].legend()

    # set x limits
    axs[0].set_xlim([1e-2,1e2])
    axs[1].set_xlim([1e-2,1e2])

    # add grid 
    axs[0].grid()
    axs[1].grid()

    # save figure
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/appendix/elong_benchmark.png', dpi=1000, bbox_inches='tight')


    plt.show()
