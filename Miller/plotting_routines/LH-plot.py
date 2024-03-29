import AEtok.AE_tokamak_calculation as AEtok
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import h5py
import matplotlib        as mpl
from matplotlib import rc
import matplotlib.ticker as ticker
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

omn     ='scan'
eta     = 0.0
epsilon = 1/3
q       = 3.0
kappa   = 3/2
delta   = np.asarray([-0.5,0.5])
dR0dr   =-0.5
s_kappa = 0.5
s_delta = 0.0
theta_res   = int(1e3+1)
L_ref       = 'major'




def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)



# Construct grid for total integral
omn_grid          =  np.linspace(0, 20, num=100, dtype='float64')


omnv,  deltav  = np.meshgrid(omn_grid,delta)
AEv            = np.empty_like(omnv)
AEv_err        = np.empty_like(omnv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list = pool.starmap(AEtok.calc_AE, [(omnv[idx],eta,epsilon,q,kappa,deltav[idx],dR0dr,4*(1 - 0.25*omnv[idx]/omn_grid[-1]) ,s_kappa,deltav[idx]/(1-deltav[idx]**2)**0.5, 2 * omnv[idx]/omn_grid[-1],theta_res,L_ref) for idx, val in np.ndenumerate(AEv)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list = np.asarray(AE_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv):
        AEv[idx]    = AE_list[list_idx]
        list_idx    = list_idx + 1

    # theta = np.linspace(0,np.pi*2,theta_res)
    # f = plt.figure(1)
    # plt.plot(1 + epsilon*np.cos(theta + np.arcsin(delta)*np.sin(theta)), epsilon*kappa*np.sin(theta))
    # plt.axis('equal')
    # plt.title("Flux surface")
    # plt.xlabel(r'$R/R_0$')
    # plt.ylabel(r'$Z/R_0$')


    # levels = np.linspace(0, 30, 25)
    # print(np.amax(AEv))
    fig = plt.figure(figsize=(6, 2.0)) #figsize=(3.375, 2.3)
    ax  = fig.gca()
    cnt1 = plt.semilogy(omnv[1,:], AEv[1,:],'blue',label=r'$\delta =+0.5$',linestyle='dashed')
    cnt0 = plt.semilogy(omnv[0,:], AEv[0,:],'red',label=r'$\delta = -0.5$',linestyle='dashdot')
    plt.legend(loc='lower right')
    # plt.title(r'Available Energy as a function of $s$ and $\alpha$' '\n' r'$\omega_n$={}, $\eta$={}, $\epsilon$={}, $q$={}, $d R_0/ d r$={}, $\kappa$={}, $s_\kappa$={}, $\delta$={}, $s_\delta$={}' '\n'.format(omn,eta,epsilon,q,dR0dr,kappa,s_kappa,delta,s_delta))
    plt.xlabel(r'$\hat{\omega}_n$')
    plt.ylabel(r'$\widehat{A}$')
    # plt.text(3.2, -1.6, r'$(b)$',c='white')
    ax.xaxis.set_tick_params(which='major', direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', direction='in', top='on')
    ax.yaxis.set_tick_params(which='minor', direction='in', top='on')
    ax.grid(True)
    aemax=np.amax(AEv)
    ax.text(1,25/30*aemax,r'$(c)$',horizontalalignment='center',verticalalignment='center')
    plt.tight_layout()
    # plt.subplots_adjust(left=0.15, right=0.88, top=0.96, bottom=0.14)
    plt.margins(0.1)
    plt.xlim((omn_grid.min(),omn_grid.max()))
    plt.ylim(1e-2,1e1)
    #plt.yscale('log')
    #plt.ylim(bottom=0.01)
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/LH/line_plot.png', format='png',
                #This is recommendation for publication plots
                dpi=2000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight', pad_inches = 0.1)
    plt.show()