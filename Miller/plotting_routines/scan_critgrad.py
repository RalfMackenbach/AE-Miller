import AEtok.AE_tokamak_calculation as AEtok
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import matplotlib        as mpl
from matplotlib import rc
import matplotlib.ticker as ticker
from    scipy.signal        import  savgol_filter
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

omn     = 'scan'
res     = 100
eta     = 0.0
epsilon = 1e-4
q       = 2.0
kappa   = 1.5
delta   = 0.5
dR0dr   = 0.0
s_q     = 0.0
s_kappa = 0.0
s_delta = 0.0
alpha   = 0.0
theta_res   = int(1e3+1)
L_ref       = 'major'





def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)



# Construct grid for total integral
omn_grid       =  np.logspace(-3.0, +3.0, num=res)


AEv            = np.empty_like(omn_grid)
AEv_err        = np.empty_like(omn_grid)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list = pool.starmap(AEtok.calc_AE, [(omn_grid[idx],eta,epsilon,q,kappa,delta,dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,L_ref) for idx, val in np.ndenumerate(omn_grid)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list = np.asarray(AE_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv):
        AEv[idx]    = AE_list[list_idx]
        list_idx    = list_idx + 1


    # plt.loglog(omn_grid,AE_list)
    # plt.loglog(omn_grid,AE_list[0]*(omn_grid/omn_grid[0])**(7/2),linestyle='dotted',color='black')
    # plt.loglog(omn_grid,AE_list[-1]*(omn_grid/omn_grid[-1])**(1),linestyle='dotted',color='black')
    # plt.axvline(x = crit_grad,color = 'r', linestyle='dashed')
    # plt.ylabel('available energy')
    # plt.xlabel('omn')
    fig, axs = plt.subplots(1,2, figsize=(6.850394, 5.0/2))
    axs[0].loglog(omn_grid,AE_list,label=r'$\widehat{A}$')
    axs[0].loglog(omn_grid[0:int(res/2)],(omn_grid[0:int(res/2)]/omn_grid[0])**(3.0)*AE_list[0]*3.0,linestyle='dashed',label=r'$\hat{\omega}_n^3$')
    axs[0].loglog(omn_grid[int(res/2)::],(omn_grid[int(res/2)::]/omn_grid[-1])**(1.0)*AE_list[-1]*3.0,linestyle='dashdot',label=r'$\hat{\omega}_n$',color='black')
    axs[0].legend(loc='upper left')
    axs[0].set_xlabel(r'$\hat{\omega}_n$')
    axs[0].set_ylabel(r'$\widehat{A}$')
    axs[0].set_xlim((1e-3,1e3))
    axs[1].set_ylim((AE_list[0],AE_list[-1]))
    axs[1].plot(omn_grid,AE_list)
    p=np.polyfit(omn_grid[-10::], AE_list[-10::], 1)
    axs[1].plot(omn_grid,p[0]*omn_grid + p[1],linestyle='dotted',color='black')
    print('crit grad is:', -p[1]/p[0])
    aeomnmax = AEtok.calc_AE(5.0,eta,epsilon,q,kappa,delta,dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,lam_res,del_sign,L_ref)
    axs[1].axvline(x=-p[1]/p[0], color='red', linestyle='solid')
    axs[1].text(-p[1]/p[0]-0.5, aeomnmax/2, r'$\hat{\omega}_{c}$',rotation='vertical',color='red')
    axs[1].set_xlabel(r'$\hat{\omega}_n$')
    axs[1].set_ylabel(r'$\widehat{A}$')
    axs[1].set_ylim((0.0, aeomnmax))
    axs[1].set_xlim((0.0, 5.0))
    axs[0].minorticks_off()
    axs[1].text(5.0*0.8, aeomnmax/7, r'$(b)$')
    logval = (np.log10(AE_list[-1])-np.log10(AE_list[0]))/10 + np.log10(AE_list[0])
    axs[0].text(1e2, 10**logval, r'$(a)$')
    plt.tight_layout()
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/crit_grad/example.eps', format='eps',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    plt.show()
