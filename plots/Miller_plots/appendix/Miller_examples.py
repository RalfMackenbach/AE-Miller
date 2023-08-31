import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


# define Miller cross-section
def miller_cross_section(epsilon,kappa,delta):

    # create theta array
    theta = np.linspace(0,2*np.pi,1000)

    # Create R values
    R = 1.0 + epsilon * np.cos(theta + np.arcsin(delta) * np.sin(theta))
    Z = kappa * epsilon * np.sin(theta )

    return R,Z

# plot a plot (kappa,delta)
# on a 1x4 grid
kappa_small = 2/3
kappa_large = 3/2
delta_val = 0.9
epsilon = 1/3 # for all plots

# make plots
fig, ax = plt.subplots(1,4,figsize=(6.850394, 2.5),sharex=True,sharey=True,tight_layout=True)


# plot (kappa,delta) = (kappa_small,delta)
kappa   = kappa_small
delta   = delta_val
R,Z = miller_cross_section(epsilon,kappa,delta)
ax[0].plot(R,Z)
ax[0].set_xlabel(r'$R$')
ax[0].set_ylabel(r'$Z$')


# plot (kappa,delta) = (kappa_small,-delta)
kappa   =  kappa_small
delta   = -delta_val
R,Z = miller_cross_section(epsilon,kappa,delta)
ax[1].plot(R,Z)
ax[1].set_xlabel(r'$R$')

# plot (kappa,delta) = (kappa_large,delta)
kappa   = kappa_large
delta   = delta_val
R,Z = miller_cross_section(epsilon,kappa,delta)
ax[2].plot(R,Z)
ax[2].set_xlabel(r'$R$')

# plot (kappa,delta) = (kappa_large,-delta)
kappa   =  kappa_large
delta   = -delta_val
R,Z = miller_cross_section(epsilon,kappa,delta)
ax[3].plot(R,Z)
ax[3].set_xlabel(r'$R$')

# set aspect ratio
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[2].set_aspect('equal')
ax[3].set_aspect('equal')

# save plot as png dpi=1000
plt.savefig('Miller_examples.png',dpi=1000)

# show plot
plt.show()