import AEtok.AE_tokamak_calculation as AEtok
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import h5py
import matplotlib        as mpl
from matplotlib import rc
import matplotlib.ticker as ticker
import scipy
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

omn     = 1.0
eta     = 0.0
epsilon = 1/3
q       = 3.0
dR0dr   =-0.5
s_kappa = 0.0
s_delta = 0.0
theta_res   = int(1e3+1)
L_ref       = 'minor'
A = 3.0
rho = 1.0
s_min = -1.0
s_max = +4.0
alpha_min = 0.0
alpha_max = 1.0
kappa_min = 0.5
kappa_max = 2.0
delta_min =-0.5
delta_max = 0.5
res = 2

s_grid          =   np.linspace(s_min, s_max, num=res, dtype='float64')
alpha_grid      =   np.linspace(alpha_min, alpha_max, num=res, dtype='float64')
sv, alphav      =   np.meshgrid(s_grid, alpha_grid, indexing='ij')
delta_opt       =   np.empty_like(sv)
kappa_opt       =   np.empty_like(sv)
AE_opt          =   np.empty_like(sv)



def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)

def opt_func(s_q,alpha):
    fun = lambda x: -1*AEtok.calc_AE(omn,eta,epsilon,  q,x[0], x[1],dR0dr, s_q,s_kappa,s_delta,alpha,theta_res,L_ref,A,rho)
    res = scipy.optimize.shgo(fun,bounds=((kappa_min,kappa_max),(delta_min,delta_max)),options={'disp': False} )
    return res

if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list = pool.starmap(opt_func, [(sv[idx],alphav[idx]) for idx, val in np.ndenumerate(sv)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))
    pool.close()

    AE_list = np.asarray(AE_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(sv):
        vals              = (AE_list[list_idx]).x
        fun               = (AE_list[list_idx]).fun
        kappa_opt[idx]    = vals[0]
        delta_opt[idx]    = vals[1]
        AE_opt[idx]       = np.abs(fun)
        list_idx = list_idx+1

    fig, axs = plt.subplots(1,4, figsize=(6.850394, 5.0/2)) #figsize=(6.850394, 3.0)
    pAE         = axs[0].pcolor(alphav,sv,AE_opt,cmap='plasma')
    pAE.set_edgecolor('face')
    cbarae   = fig.colorbar(-1*pAE, ticks=[np.amin(AE_opt),np.amax(AE_opt)], ax=axs[0],orientation="horizontal",pad=0.3,label=r'$\widehat{A}$')
    cbarae.set_ticks(ticks=[np.amin(AE_opt),np.amax(AE_opt)])
    cbarae.set_ticklabels([fmt(np.amin(-1*AE_opt),1),fmt(np.amax(-1*AE_opt),1)])
    pkappa      = axs[1].pcolor(alphav,sv,kappa_opt,cmap='viridis')
    pkappa.set_edgecolor('face')
    cbarkappa   = fig.colorbar(pkappa, ticks=[kappa_min,kappa_max], ax=axs[1],orientation="horizontal",pad=0.3,label=r'$\kappa$')
    pdelta      = axs[2].pcolor(alphav,sv,delta_opt,cmap='viridis')
    pdelta.set_edgecolor('face')
    cbardelta   = fig.colorbar(pdelta, ticks=[delta_min,delta_max], ax=axs[2],orientation="horizontal",pad=0.3,label=r'$\delta$')
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import BoundaryNorm
    levels =    MaxNLocator(nbins=4).tick_values(0, 4)
    cmap = plt.colormaps['jet']
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    cat         = 2+2*np.arctan2(delta_opt,kappa_opt-1)/np.pi
    pcat        = axs[3].pcolor(alphav,sv,cat,cmap=cmap,norm=norm)
    pcat.set_edgecolor('face')
    cbarcat     = fig.colorbar(pcat, ax=axs[3],orientation="horizontal",pad=0.3)
    cbarcat.set_ticks(ticks=[0.5,1.5,2.5,3.5],labels=[r'$\mathrm{NC}$',r'$\mathrm{NT}$',r'$\mathrm{PT}$',r'$\mathrm{PC}$'])

    axs[0].set_ylabel(r'$s$')
    axs[0].set_xlabel(r'$\alpha$')
    axs[1].set_xlabel(r'$\alpha$')
    axs[2].set_xlabel(r'$\alpha$')
    axs[3].set_xlabel(r'$\alpha$')
    axs[0].set_yticks([-1,0,1,2,3,4])
    axs[1].set_yticks([-1,0,1,2,3,4])
    axs[2].set_yticks([-1,0,1,2,3,4])
    axs[3].set_yticks([-1,0,1,2,3,4])
    axs[0].set_yticklabels([r'$-1$',r'$0$',r'$1$',r'$2$',r'$3$',r'$4$'])
    axs[1].set_yticklabels([None,None,None,None,None,None])
    axs[2].set_yticklabels([None,None,None,None,None,None])
    axs[3].set_yticklabels([None,None,None,None,None,None])
    axs[0].text(0.8,-0.5,r'$(a)$',color='white')
    axs[1].text(0.8,-0.5,r'$(b)$')
    axs[2].text(0.8,-0.5,r'$(c)$')
    axs[3].text(0.8,-0.5,r'$(d)$')
    plt.tight_layout()
    plt.savefig('/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/optimisation/optimized-triptych.eps', format='eps',
                #This is recommendation for publication plots
                dpi=1000,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight')
    plt.show()


