# here, we plot the bounce-averaged drifts for s-alpha tokamaks
# latex font
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from matplotlib import rc
rc('text', usetex=True)


def CHM(k2):
    '''
    ``CHM`` Contains the analytical bounce-averaged drift
    of Connor, Hastie, and Martin, for a s-alpha tokamak.
    '''
    E   = ellipe(k2, out=np.zeros_like(k2))
    K   = ellipk(k2, out=np.zeros_like(k2))
    EK  = E/K
    G1 = EK - 1/2
    G2 = EK + k2 - 1
    G3 = 2/3 * (EK * (2 * k2 - 1) +1 - k2 )
    return G1, G2, G3

# now plot all three drifts as a function of k2 
# solid, dashed, dotted
scale=1/2
fig, ax = plt.subplots(1,1, figsize=(scale*6.850394, scale*5.0))
k2 = np.linspace(0,1,int(1e6))
G1, G2, G3 = CHM(k2)
plt.plot(k2, G1, 'k-', label=r'$G_1$')
plt.plot(k2, G2, 'k--', label=r'$G_2$')
plt.plot(k2, G3, 'k:', label=r'$G_3$')
# add labels
plt.xlabel(r'$k^2$')
plt.ylabel(r'$G_i$')
plt.xlim([0,1])
plt.ylim([-0.6,0.6])
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('CHM_drifts.png',dpi=1000)
plt.show()