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
from    scipy.signal        import  savgol_filter
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

omn_fac = 1e2
omn1 = omn_fac*1.0
omn2 = omn_fac*5.0
omn3 = omn_fac*10.0
eta     = 0.0
epsilon = 1/3
q       = 3.0
kappa   = 1.0
delta   = 0.0
dR0dr   = 0.0
s_q     = 0.0
s_kappa = 0.0
s_delta = 0.0
alpha   = 0.0
theta_res   = int(1e2 +1)
lam_res     = int(1e2)
del_sign    = 0.0
L_ref       = 'major'





def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)



s_grid          =  np.linspace(-1.0, +4.0, num=10, dtype='float64')
alpha_grid      =  np.linspace( 0.0, +1.0, num=10, dtype='float64')


sv, alphav = np.meshgrid(s_grid, alpha_grid, indexing='ij')
AEv1           = np.empty_like(sv)
AEv2           = np.empty_like(sv)
AEv3           = np.empty_like(sv)
crit_grad      = np.empty_like(sv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    var_arr = np.linspace(0.0,4.0,10)
    crit_grad_arr = np.empty_like(var_arr)

    # time the full integral
    start_time = time.time()
    AE_list_1 = pool.starmap(AEtok.calc_AE, [(omn1,eta,epsilon,q,kappa,delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(AEv1)])
    AE_list_2 = pool.starmap(AEtok.calc_AE, [(omn2,eta,epsilon,q,kappa,delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(AEv2)])
    AE_list_3 = pool.starmap(AEtok.calc_AE, [(omn3,eta,epsilon,q,kappa,delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(AEv3)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))
    
    list_idx = 0
    for idx, val in np.ndenumerate(AEv1):
        AEv1[idx]   = AE_list_1[list_idx]
        list_idx    = list_idx + 1

    list_idx = 0
    for idx, val in np.ndenumerate(AEv2):
        AEv2[idx]   = AE_list_2[list_idx]
        list_idx    = list_idx + 1
    
    list_idx = 0
    for idx, val in np.ndenumerate(AEv3):
        AEv3[idx]   = AE_list_3[list_idx]
        list_idx    = list_idx + 1

    pool.close()

    # fit 
    for idx, _ in np.ndenumerate(crit_grad):
        p=np.polyfit(np.asarray([omn1,omn2,omn3]), np.asarray([AEv1[idx],AEv2[idx],AEv3[idx]]), 1)
        if p[0] == 0:
            crit_grad[idx] = None
        if p[0] != 0:
            crit_grad[idx] =  -p[1]/p[0]
        

    levels_arr=np.linspace(np.nanmin(crit_grad),np.nanmax(crit_grad),20)
    plt.contourf(sv,alphav,crit_grad,levels=levels_arr,cmap='viridis_r')
    plt.colorbar(label=r'$L_{n,\mathrm{crit}}/R_0$')
    plt.xlabel(r'$s$')
    plt.ylabel(r'$\alpha$')
    plt.show()