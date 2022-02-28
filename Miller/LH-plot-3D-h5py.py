import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import h5py
import matplotlib        as mpl
import AE_tokamak_calculation as AEtok
import plotly.graph_objects as go



f = h5py.File('isocontour_eta=0.0_eps=0.3_q=2.5_kappa=1.7_delta=0.6_dR0dr=0.0_skappa=0.0_sdelta=0.0.hdf5','r')
omnv    = np.asarray(f['omnv'])
sv      = np.asarray(f['sv'])
alphav  = np.asarray(f['alphav'])
AEv     = np.asarray(f['AEv'])

fig = go.Figure(data=go.Isosurface(
x=omnv.flatten(),
y=alphav.flatten(),
z=sv.flatten(),
value=AEv.flatten(),
isomin=np.amin(AEv)+0.5,
isomax=np.amax(AEv)*0.9,
opacity=0.6,
surface_count=5, # number of isosurfaces, 2 by default: only min and max
colorbar_nticks=5, # colorbar ticks correspond to isosurface values
caps=dict(x_show=False, y_show=False, z_show=False)
))
fig.update_layout(scene = dict(
                xaxis_title='density gradient',
                yaxis_title='pressure gradient',
                zaxis_title='magnetic shear'),
                )
fig.write_html("/Users/ralfmackenbach/Documents/GitHub/AE-tok/plots/Miller plots/AE_isocont.html", include_mathjax="cdn")
import plotly.io as pio

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-1.25, y=1.25, z=1.25)
)
fig.layout.scene.camera
fig.show()
fig.update_layout(scene_camera=camera)
fig.write_image("test.pdf")
