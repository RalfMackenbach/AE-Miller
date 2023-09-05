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

omn_fac = 1e4
omn1 = omn_fac*((1+np.sqrt(5))/2)**1
omn2 = omn_fac*((1+np.sqrt(5))/2)**2
omn3 = omn_fac*((1+np.sqrt(5))/2)**3
eta     = 0.0
epsilon = 1/3
q       = 2.0
kappa   = 3/2
delta   = 0.5
dR0dr   = 0.0
s_q     = 1.0
s_kappa = 0.0
s_delta = 0.0
alpha   = 0.0
theta_res   = int(1e3+1)
L_ref       = 'major'





def fmt(x, pos):
    a = round(x,pos)
    return r'${}$'.format(a)

res = 20

s_grid          =  np.linspace(-1.0, +5.0, num=res)
alpha_grid      =  np.linspace( 0.0, +1.0, num=res)

kappa_grid      =  np.linspace(+0.5, +2.0, num=res)
delta_grid      =  np.linspace(-0.8, +0.8, num=res)


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
    AE_sa_3 = pool.starmap(AEtok.calc_AE, [(omn3,eta,epsilon,q,kappa,delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,L_ref) for idx, val in np.ndenumerate(AEsa3)])
    AE_kd_1 = pool.starmap(AEtok.calc_AE, [(omn1,eta,epsilon,q,kappav[idx],deltav[idx],dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,L_ref) for idx, val in np.ndenumerate(AEkd1)])
    AE_kd_2 = pool.starmap(AEtok.calc_AE, [(omn2,eta,epsilon,q,kappav[idx],deltav[idx],dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,L_ref) for idx, val in np.ndenumerate(AEkd2)])
    AE_kd_3 = pool.starmap(AEtok.calc_AE, [(omn3,eta,epsilon,q,kappav[idx],deltav[idx],dR0dr,s_q,s_kappa,s_delta,alpha,theta_res,L_ref) for idx, val in np.ndenumerate(AEkd3)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))
    pool.close()
    list_idx = 0
    for idx, val in np.ndenumerate(AEsa1):
        AEsa1[idx]   = AE_sa_1[list_idx]
        AEsa2[idx]   = AE_sa_2[list_idx]
        AEsa3[idx]   = AE_sa_3[list_idx]
        list_idx    = list_idx + 1

    list_idx = 0
    for idx, val in np.ndenumerate(AEkd1):
        AEkd1[idx]   = AE_kd_1[list_idx]
        AEkd2[idx]   = AE_kd_2[list_idx]
        AEkd3[idx]   = AE_kd_3[list_idx]
        list_idx    = list_idx + 1
    
    # fit 
    for idx, _ in np.ndenumerate(critgrad_sa):
        p=np.polyfit(np.asarray([omn1,omn2,omn3]), np.asarray([AEsa1[idx],AEsa2[idx],AEsa3[idx]]), 1)
        if p[0] != 0:
            critgrad_sa[idx] =  -p[1]/p[0]
        else: 
            critgrad_sa[idx] =  np.nan

    for idx, _ in np.ndenumerate(critgrad_kd):
        p=np.polyfit(np.asarray([omn1,omn2,omn3]), np.asarray([AEkd1[idx],AEkd2[idx],AEkd3[idx]]), 1)
        if p[0] != 0:
            critgrad_kd[idx] =  -p[1]/p[0]
        else: 
            critgrad_kd[idx] =  np.nan

    maxsa = np.nanmax(critgrad_sa)
    maxkd = np.nanmax(critgrad_kd)
    minsa = np.nanmin(critgrad_sa)
    minkd = np.nanmin(critgrad_kd)
    max  = np.nanmax([maxsa,maxkd])
    min  = np.nanmin([minsa,minkd])
    #print(maxsa,minsa,maxkd,minkd)
    levelssa = np.linspace(min,max,30)
    levelskd = np.linspace(min,max,30)
    plt.rcParams['figure.constrained_layout.use'] = True
    fig, axs = plt.subplots(2,1, figsize=(4.5,6.0)) #figsize=(6.850394, 3.0)
    cnt0 = axs[0].contourf(alphav, sv,      critgrad_sa, levels=levelssa, cmap='viridis_r')
    cnt1 = axs[1].contourf(kappav, deltav,  critgrad_kd, levels=levelskd, cmap='viridis_r')
    for c in cnt0.collections:
        c.set_edgecolor("face")
    for c in cnt1.collections:
        c.set_edgecolor("face")
    cbar0 = fig.colorbar(cnt0,ticks=[minsa,maxsa],ax=axs.ravel().tolist(),label=r'gradient threshold, $\hat{\omega}_c$')
    # cbar1 = fig.colorbar(cnt1,ticks=[minkd,maxkd],ax=axs[1],label=r'$\hat{\omega}_c$')
    cbar0.set_ticklabels([fmt(minsa,1),fmt(maxsa,1)])
    # cbar1.set_ticklabels([fmt(minkd,1),fmt(maxkd,1)])
    cbar0.solids.set_edgecolor("face")
    # cbar1.solids.set_edgecolor("face")
    axs[0].set_xlabel(r'pressure gradient, $\alpha$')
    axs[0].set_ylabel(r'magnetic shear, $s$')
    axs[1].set_xlabel(r'elongation, $\kappa$')
    axs[1].set_ylabel(r'triangularity, $\delta$')
    
    # set x and y values of text consistently for the differing domains
    xt = 0.12
    yt = 0.12


    x_sa = 0 + xt*1
    y_sa = -1 + 6 * xt
    x_kd = 0.5 + 1.5 * yt
    y_kd = -0.8 + 1.6 * yt

    axs[0].text(x_sa,y_sa,r'$(a)$',horizontalalignment='center',verticalalignment='center')
    axs[1].text(x_kd,y_kd,r'$(b)$',horizontalalignment='center',verticalalignment='center')

    # save figure
    plt.savefig('critgrad-plot.png',bbox_inches='tight',dpi=1000)

    plt.show()