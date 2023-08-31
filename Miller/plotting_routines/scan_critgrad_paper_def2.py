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

omn_fac = 1e2
omn1 = 1e-3
omn2 = 1e3
eta     = 0.0
epsilon = 1/3
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

res = 10

s_grid          =  np.linspace(-1.0, +4.0, num=res, dtype='float64')
alpha_grid      =  np.linspace( 0.0, +1.0, num=res, dtype='float64')

kappa_grid      =  np.linspace(+0.5, +2.0, num=res, dtype='float64')
delta_grid      =  np.linspace(-0.8, +0.8, num=res, dtype='float64')


sv, alphav = np.meshgrid(s_grid, alpha_grid, indexing='ij')
kappav, deltav = np.meshgrid(kappa_grid, delta_grid, indexing='ij')
AEsa1           = np.empty_like(sv)
AEsa2           = np.empty_like(sv)
AEsa3           = np.empty_like(sv)
AEkd1           = np.empty_like(kappav)
AEkd2           = np.empty_like(kappav)
AEkd3           = np.empty_like(kappav)
critgrad_sa     = np.empty_like(sv)
critgrad_kd     = np.empty_like(kappav)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    var_arr = np.linspace(0.0,4.0,10)
    crit_grad_arr = np.empty_like(var_arr)

    # time the full integral
    start_time = time.time()
    AE_sa_1 = pool.starmap(AEtok.calc_AE, [(omn1,eta,epsilon,q,kappa,delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,L_ref) for idx, val in np.ndenumerate(AEsa1)])
    AE_sa_2 = pool.starmap(AEtok.calc_AE, [(omn2,eta,epsilon,q,kappa,delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,L_ref) for idx, val in np.ndenumerate(AEsa2)])
    AE_kd_1 = pool.starmap(AEtok.calc_AE, [(omn1,eta,epsilon,q,kappav[idx],deltav[idx],dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,L_ref) for idx, val in np.ndenumerate(AEkd1)])
    AE_kd_2 = pool.starmap(AEtok.calc_AE, [(omn2,eta,epsilon,q,kappav[idx],deltav[idx],dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,L_ref) for idx, val in np.ndenumerate(AEkd2)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))
    
    list_idx = 0
    for idx, val in np.ndenumerate(AEsa1):
        AEsa1[idx]   = AE_sa_1[list_idx]
        AEsa2[idx]   = AE_sa_2[list_idx]
        list_idx    = list_idx + 1

    list_idx = 0
    for idx, val in np.ndenumerate(AEkd1):
        AEkd1[idx]   = AE_kd_1[list_idx]
        AEkd2[idx]   = AE_kd_2[list_idx]
        list_idx    = list_idx + 1
    
    # fit 
    for idx, _ in np.ndenumerate(critgrad_sa):
        intersection = np.sqrt(AEsa2[idx]/AEsa1[idx])*np.sqrt(omn1/omn2)*omn1
        critgrad_sa[idx] =  intersection

    for idx, _ in np.ndenumerate(critgrad_kd):
        intersection = np.sqrt(AEkd2[idx]/AEkd1[idx])*np.sqrt(omn1/omn2)*omn1
        critgrad_kd[idx] =  intersection

    pool.close()

    maxsa = np.amax(critgrad_sa)
    maxkd = np.amax(critgrad_kd)
    minsa = np.amin(critgrad_sa)
    minkd = np.amin(critgrad_kd)
    max  = np.amax([maxsa,maxkd])
    min  = np.amin([minsa,minkd])
    levelssa = np.linspace(minsa,maxsa,100)
    levelskd = np.linspace(minkd,maxkd,100)
    fig, axs = plt.subplots(1,2, figsize=(6.850394, 5.0/2)) #figsize=(6.850394, 3.0)
    cnt0 = axs[0].contourf(alphav, sv,      critgrad_sa, levels=levelssa, cmap='viridis_r')
    cnt1 = axs[1].contourf(kappav, deltav,  critgrad_kd, levels=levelskd, cmap='viridis_r')
    for c in cnt0.collections:
        c.set_edgecolor("face")
    for c in cnt1.collections:
        c.set_edgecolor("face")
    cbar0 = fig.colorbar(cnt0,ticks=[minsa,maxsa],ax=axs[0])
    cbar1 = fig.colorbar(cnt1,ticks=[minkd, maxkd],ax=axs[1],label=r'$\hat{\omega}_c$')
    cbar0.set_ticklabels([fmt(minsa,1),fmt(maxsa,1)])
    cbar1.set_ticklabels([fmt(minkd,1),fmt(maxkd,1)])
    cbar0.solids.set_edgecolor("face")
    cbar1.solids.set_edgecolor("face")
    axs[0].set_xlabel(r'$\alpha$')
    axs[0].set_ylabel(r'$s$')
    axs[1].set_xlabel(r'$\kappa$')
    axs[1].set_ylabel(r'$\delta$')

    axs[0].text(1/8, -0.3, r'$(a)$',ha='center', va='center')
    axs[1].text((2-0.5)/8+0.5, (1.6)/8+-0.8, r'$(b)$',ha='center', va='center')
    plt.tight_layout()
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/crit-grad/contours_critgrad.eps', format='eps',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    plt.show()