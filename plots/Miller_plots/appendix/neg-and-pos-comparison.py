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

# return curvature
def curvature(R,Z):
    Z_p  = np.gradient(Z)
    Z_pp = np.gradient(Z_p)
    R_p  = np.gradient(R)
    R_pp = np.gradient(R_p)
    return np.abs(Z_p * R_pp - Z_pp * R_p)/np.power(Z_p**2 + R_p**2,3/2)

# plot a negative and positive triangularity Miller cross-section side by side
# define Miller cross-section
epsilon = 1/3
kappa   = 1.5
delta   = 0.8


# first plot positive triangularity
R,Z = miller_cross_section(epsilon,kappa,delta)

# plot negative triangularity
R_neg,Z_neg = miller_cross_section(epsilon,kappa,-delta)

# plot
fig, ax = plt.subplots(1,2,figsize=(6.850394, 3.5),sharex=True,sharey=True)

ax[0].plot(R,Z)
ax[0].set_xlabel(r'$R$')
ax[0].set_ylabel(r'$Z$')
ax[0].set_title('Positive triangularity')

ax[1].plot(R_neg,Z_neg)
ax[1].set_xlabel(r'$R$')
ax[1].set_ylabel(r'$Z$')
ax[1].set_title('Negative triangularity')

# set aspect ratio
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')

# add arrow pointing to region of strongest curvature

# first find the index of the strongest curvature for positive triangularity
cur = curvature(R,Z)
idx_pos = np.argmax(cur)

# next find the index of the strongest curvature for negative triangularity
cur_neg = curvature(R_neg,Z_neg)
idx_neg = np.argmax(cur_neg)



# first plot positive triangularity
# arrows at + and - Z
ax[0].annotate('',xy=(R[idx_pos], np.abs(Z[idx_pos])), xytext=(1.0, 0.1),
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
ax[0].annotate('',xy=(R[idx_pos], -np.abs(Z[idx_pos])), xytext=(1.0,-0.1),
            arrowprops=dict(facecolor='green', shrink=0.05),
            )



# next plot negative triangularity
# arrows at + and - Z
# text centering to the left of the arrow
ax[1].annotate('',xy=(R_neg[idx_neg], np.abs(Z_neg[idx_neg])), xytext=(1.0, 0.1),
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
ax[1].annotate('',xy=(R_neg[idx_neg], -np.abs(Z_neg[idx_neg])), xytext=(1.0,-0.1),
            arrowprops=dict(facecolor='red', shrink=0.05),
            )

# now add text at the end of the arrow
ax[0].annotate(r'$|R_c|^{-1} \gg 1$',xy=(1.0, 0.0), xytext=(1.0, 0.0),size=12,ha='center',va='center')
ax[1].annotate(r'$|R_c|^{-1} \gg 1$',xy=(1.0, 0.0), xytext=(1.0, 0.0),size=12,ha='center',va='center')


plt.tight_layout()

# save as png, dpi=1000
plt.savefig('neg-and-pos-comparison.png',dpi=1000)

plt.show()