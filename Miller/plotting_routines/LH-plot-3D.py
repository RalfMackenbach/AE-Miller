import sys
sys.path.insert(1, '/Users/ralfmackenbach/Documents/GitHub/AE-tok/Miller/scripts')
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import h5py
import matplotlib        as mpl
import AE_tokamak_calculation as AEtok
import plotly.graph_objects as go


omn     = 'scan'
eta     = 0.0
epsilon = 0.3
q       = 3.0
kappa   = 2.0
delta   = 0.7
dR0dr   = 0.0
s_kappa = 0.0
s_delta = 0.0
theta_res   = int(1e2 +1)
lam_res     = int(1e2)
del_sign    = 0.0
L_ref       = 'major'





def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)

def plot_contour3d(self, **kwargs):
        '''
        use mayavi.mlab to plot 3d contour.

        Parameter
        ---------
        kwargs: {
            'maxct'   : float,max contour number,
            'nct'     : int, number of contours,
            'opacity' : float, opacity of contour,
            'widths'   : tuple of int
                        number of replication on x, y, z axis,
        }
        '''
        if not mayavi_installed:
            self.__logger.warning("Mayavi is not installed on your device.")
            return
        # set parameters
        widths = kwargs['widths'] if 'widths' in kwargs else (1, 1, 1)
        elf_data, grid = self.expand_data(self.elf_data, self.grid, widths)
#        import pdb; pdb.set_trace()
        maxdata = np.max(elf_data)
        maxct = kwargs['maxct'] if 'maxct' in kwargs else maxdata
        # check maxct
        if maxct > maxdata:
            self.__logger.warning("maxct is larger than %f", maxdata)
        opacity = kwargs['opacity'] if 'opacity' in kwargs else 0.6
        nct = kwargs['nct'] if 'nct' in kwargs else 5
        # plot surface
        surface = mlab.contour3d(elf_data)
        # set surface attrs
        surface.actor.property.opacity = opacity
        surface.contour.maximum_contour = maxct
        surface.contour.number_of_contours = nct
        # reverse axes labels
        mlab.axes(xlabel='z', ylabel='y', zlabel='x')
        mlab.outline()
        mlab.show()

        return



# Construct grid for total integral
omn_grid          =  np.linspace(+1.0,+10.0, num=20, dtype='float64')
s_grid            =  np.linspace(+0.0, +2.0, num=20, dtype='float64')
alpha_grid        =  np.linspace(+0.0, +2.0, num=20, dtype='float64')


omnv, sv, alphav  = np.meshgrid(omn_grid,s_grid,alpha_grid)
AEv               = np.empty_like(omnv)
AEv_err           = np.empty_like(omnv)


if __name__ == "__main__":

    pool = mp.Pool(mp.cpu_count())
    print('Number of cores used: {}'.format(mp.cpu_count()))

    # time the full integral
    start_time = time.time()
    AE_list = pool.starmap(AEtok.calc_AE, [(omnv[idx],eta,epsilon,q,kappa,delta,dR0dr,sv[idx],s_kappa,s_delta,alphav[idx],theta_res,lam_res,del_sign,L_ref) for idx, val in np.ndenumerate(AEv)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))

    pool.close()

    AE_list = np.asarray(AE_list)

    # reorder data full int
    list_idx = 0
    for idx, val in np.ndenumerate(AEv):
        AEv[idx]    = AE_list[list_idx]
        list_idx    = list_idx + 1


    hf      = h5py.File("/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/LH/isocontour_eta={}_eps={}_q={}_kappa={}_delta={}_dR0dr={}_skappa={}_sdelta={}.hdf5".format(eta,epsilon,q,kappa,delta,dR0dr,s_kappa,s_delta), "w")
    hf.create_dataset('omnv',   data=omnv)
    hf.create_dataset('sv',     data=sv)
    hf.create_dataset('alphav', data=alphav)
    hf.create_dataset('AEv',    data=AEv)
    hf.close()

    fig = go.Figure(data=go.Isosurface(
    x=omnv.flatten(),
    y=alphav.flatten(),
    z=sv.flatten(),
    value=AEv.flatten(),
    isomin=np.amin(AEv)+0.2,
    isomax=np.amax(AEv),
    surface_count=5, # number of isosurfaces, 2 by default: only min and max
    colorbar_nticks=5, # colorbar ticks correspond to isosurface values
    caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    fig.update_layout(scene = dict(
                    xaxis_title=r'$\omega_n$',
                    yaxis_title=r'$\alpha$',
                    zaxis_title=r'$s$'),
                    )
    fig.write_html("/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller_plots/LH/fig.html")
