import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json 
import numpy.ma as ma
import matplotlib.colors as colors
import AEtok.AE_tokamak_calculation as AEtok
plt.close('all')

mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)

# PRL constant of proportionality
C = 1072.6914449397775

# open TGLF data
f1 = open("data/tglf_ae_delta-kappa_DELTA_LOC_KAPPA_LOC_20x20_SAT=2_NBASIS=6_KYMAX=1_UNITS=GENE_epsilon=0.33_rmin=0.5_omn=2.0_eta=1.0_q=2.json")
f2 = open("data/tglf_ae_delta-kappa_DELTA_LOC_KAPPA_LOC_20x20_SAT=2_NBASIS=6_KYMAX=1_UNITS=GENE_epsilon=0.33_rmin=0.5_omn=4.0_eta=1.0_q=2.json")
f3 = open("data/tglf_ae_s-alpha_Q_PRIME_LOC_P_PRIME_LOC_20x20_SAT=2_NBASIS=6_KYMAX=1_UNITS=GENE_epsilon=0.33_rmin=0.5_omn=4.0_eta=1.0_q=2.json")

# now load the data
data1 = json.load(f1)
data2 = json.load(f2)
data3 = json.load(f3)


# first load the kappa delta data from first data set
kappa1 = np.array(data1['KAPPA_LOC'])
delta1 = np.array(data1['DELTA_LOC'])
Qe1    = np.array(data1['Qe'])
Qe1_mask= np.array(data1['Qe_filter'])

# now load the kappa delta data from second data set
kappa2 = np.array(data2['KAPPA_LOC'])
delta2 = np.array(data2['DELTA_LOC'])
Qe2    = np.array(data2['Qe'])
Qe2_mask= np.array(data2['Qe_filter'])

# now load s alpha data from third data set
s3     = np.array(data3['Q_PRIME_LOC'])
alpha3 = np.array(data3['P_PRIME_LOC'])
Qe3    = np.array(data3['Qe'])
Qe3_mask= np.array(data3['Qe_filter'])

# convert all lists to numpy arrays
kappa1  = np.asarray(kappa1)
delta1  = np.asarray(delta1)
Qe1     = np.asarray(Qe1)
Qe1_mask= np.asarray(Qe1_mask)
kappa2  = np.asarray(kappa2)
delta2  = np.asarray(delta2)
Qe2     = np.asarray(Qe2)
Qe2_mask= np.asarray(Qe2_mask)
s3      = np.asarray(s3)
alpha3  = np.asarray(alpha3)
Qe3     = np.asarray(Qe3)
Qe3_mask= np.asarray(Qe3_mask)

# now mask the Qe data
Qe1_masked = ma.masked_where(Qe1_mask == 0, Qe1)
Qe2_masked = ma.masked_where(Qe2_mask == 0, Qe2)
Qe3_masked = ma.masked_where(Qe3_mask == 0, Qe3)

# find the number of data points
N1=ma.MaskedArray.count(Qe1_masked)
N2=ma.MaskedArray.count(Qe2_masked)
N3=ma.MaskedArray.count(Qe3_masked)
N=N1+N2+N3


# retrieve minimal and maximal values for plotting
Qe1_min = np.min(Qe1_masked)
Qe1_max = np.max(Qe1_masked)
Qe2_min = np.min(Qe2_masked)
Qe2_max = np.max(Qe2_masked)
Qe3_min = np.min(Qe3_masked)
Qe3_max = np.max(Qe3_masked)
# print maximal values
print('Qe1_max = ', Qe1_max)
print('Qe2_max = ', Qe2_max)
print('Qe3_max = ', Qe3_max)

# next import the AE data from hdf5 file
# first the kappa delta
AEkd        = h5py.File('data/AE_data_kd_eta=1.0.hdf5', 'r')
AEkd_kap    = AEkd['kappa']
AEkd_del    = AEkd['delta']
AE1         = np.asarray(AEkd['AEkd'])
AE2         = np.asarray(AEkd['AEkd_2'])
QAE1        = C * AE1**(3/2)
QAE2        = C * AE2**(3/2)

# now same for s alpha
AEsa        = h5py.File('data/AE_data_salpha_eta=1.0.hdf5', 'r')
AEsa_s      = AEsa['shear']
AEsa_a      = AEsa['alpha']
AE3         = np.asarray(AEsa['AEsa'])
QAE3        = C * AE3**(3/2)

# # do the same for eta=0 data
# # first the kappa delta
# AEkd_eta0   = h5py.File('data/AE_data_kd_eta=0.0.hdf5', 'r')
# AEkd_kap_eta0= AEkd_eta0['kappa']
# AEkd_del_eta0= AEkd_eta0['delta']
# AE1_eta0    = np.asarray(AEkd_eta0['AEkd'])
# AE2_eta0    = np.asarray(AEkd_eta0['AEkd_2'])
# QAE1_eta0   = AE1_eta0**(3/2)
# QAE2_eta0   = AE2_eta0**(3/2)

# # now same for s alpha
# AEsa_eta0   = h5py.File('data/AE_data_salpha_eta=0.0.hdf5', 'r')
# AEsa_s_eta0 = AEsa_eta0['shear']
# AEsa_a_eta0 = AEsa_eta0['alpha']
# AE3_eta0    = np.asarray(AEsa_eta0['AEsa'])
# QAE3_eta0   = AE3_eta0**(3/2)

# # make new estimate of heat flux 
# QAE1    = (AE1_eta0 + AE1)**(3/2)
# QAE2    = (AE2_eta0 + AE2)**(3/2)
# QAE3    = (AE3_eta0 + AE3)**(3/2)


# retrieve minimal and maximal values for plotting
QAE1_min = np.min(QAE1)
QAE1_max = np.max(QAE1)
QAE2_min = np.min(QAE2)
QAE2_max = np.max(QAE2)
QAE3_min = np.min(QAE3)
QAE3_max = np.max(QAE3)


# TGLFs alpha and s are different from AEtok, so we construct them manually
alpha = np.linspace(+0., +2., 20)
s     = np.linspace(-1., +5., 20)
# meshgrid for plotting
alphav, sv = np.meshgrid(alpha, s)



# set colormap 
cmap = mpl.cm.plasma




# finally retrieve max and min values of Qe and QAE per kd and sa
Q1_min = np.log10(np.min([Qe1_min,QAE1_min]))
Q1_max = np.log10(np.max([Qe1_max,QAE1_max]))
Q2_min = np.log10(np.min([Qe2_min,QAE2_min]))
Q2_max = np.log10(np.max([Qe2_max,QAE2_max]))
Q3_min = np.log10(np.min([Qe3_min,QAE3_min]))
Q3_max = np.log10(np.max([Qe3_max,QAE3_max]))







def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    if b != 0:
        return r'${} \cdot 10^{{{}}}$'.format(a, b)
    if b == 0:
        return r'${}$'.format(a)


fig, axs = plt.subplots(2,3, figsize=(6.850394, 5.0),constrained_layout=True)

# plot TGLF data first with pcolor 
TGLF1 = axs[0,0].pcolor(kappa1,delta1,np.log10(Qe1_masked),cmap=cmap,vmin=Q1_min, vmax=Q1_max)
TGLF2 = axs[0,1].pcolor(kappa2,delta2,np.log10(Qe2_masked),cmap=cmap,vmin=Q2_min, vmax=Q2_max)
TGLF3 = axs[0,2].pcolor(alphav,sv,np.log10(Qe3_masked),cmap=cmap,vmin=Q3_min, vmax=Q3_max)

# get rid of white lines
TGLF1.set_edgecolor('face')
TGLF2.set_edgecolor('face')
TGLF3.set_edgecolor('face')

# make colorbars at top of the plot, label Q_e
cbarTGLF1 = fig.colorbar(TGLF1,ax=axs[0,0],orientation='horizontal',label=r'$\log_{10} \widehat{Q}_e$',location='top')
cbarTGLF2 = fig.colorbar(TGLF2,ax=axs[0,1],orientation='horizontal',label=r'$\log_{10} \widehat{Q}_e$',location='top')
cbarTGLF3 = fig.colorbar(TGLF3,ax=axs[0,2],orientation='horizontal',label=r'$\log_{10} \widehat{Q}_e$',location='top')

# set labels
axs[0,0].set_ylabel(r'$\delta$')
axs[0,0].set_xlabel(r'$\kappa$')
axs[0,1].set_ylabel(r'$\delta$')
axs[0,1].set_xlabel(r'$\kappa$')
axs[0,2].set_ylabel(r'$s$')
axs[0,2].set_xlabel(r'$\alpha$')

# now plot AE data with pcolor
AE1 = axs[1,0].pcolor(kappa1,delta1,np.log10(QAE1),cmap=cmap,vmin=Q1_min, vmax=Q1_max)
AE2 = axs[1,1].pcolor(kappa2,delta2,np.log10(QAE2),cmap=cmap,vmin=Q2_min, vmax=Q2_max)
AE3 = axs[1,2].pcolor(alphav,sv,np.transpose(np.log10(QAE3)),cmap=cmap,vmin=Q3_min, vmax=Q3_max)

# get rid of white lines
AE1.set_edgecolor('face')
AE2.set_edgecolor('face')
AE3.set_edgecolor('face')

# make colorbars at top of the plot, label Q_{A}
cbarAE1 = fig.colorbar(AE1,ax=axs[1,0],orientation='horizontal',label=r'$\log_{10}\widehat{Q}_{A}$',location='bottom')
cbarAE2 = fig.colorbar(AE2,ax=axs[1,1],orientation='horizontal',label=r'$\log_{10}\widehat{Q}_{A}$',location='bottom')
cbarAE3 = fig.colorbar(AE3,ax=axs[1,2],orientation='horizontal',label=r'$\log_{10}\widehat{Q}_{A}$',location='bottom')

# put x ticks on the top of AE plots
axs[1,0].xaxis.tick_top()
axs[1,1].xaxis.tick_top()
axs[1,2].xaxis.tick_top()

# set y labels alone
axs[1,0].set_ylabel(r'$\delta$')
axs[1,1].set_ylabel(r'$\delta$')
axs[1,2].set_ylabel(r'$s$')

# save figure
plt.savefig('Qe_QAE_TGLF.png',dpi=1000)





plt.show()
plt.close()

# now make a scatter plot of Q_e vs Q_{A}
plt.figure(figsize=(6.850394/2, 6.850394/3),constrained_layout=True)

plt.grid()
alpha_val = 0.2
s_val = 10
plt.scatter(QAE1,Qe1_masked,marker='.',color='k',alpha=alpha_val,s=s_val)
plt.scatter(QAE2,Qe2_masked,marker='.',color='k',alpha=alpha_val,s=s_val)
plt.scatter(np.transpose(QAE3),Qe3_masked,marker='.',color='k',alpha=alpha_val,s=s_val,label=r'$\textsc{tglf}$')

# add scatter from PRL as well
A_prl = [0.02359672, 0.08649891, 0.15689829, 0.2280691, 0.00594162, 0.01342324, 0.02091065, 0.02839889, 0.00343596, 0.00734704, 0.01126644, 0.01518755, 0.0105219, 0.01319848, 0.19905833] 
Q_prl = [7.64584715, 37.11184344, 66.28563923, 88.89918002, 0.22148522, 0.82119839, 1.84836458, 8.27240125, 0.42191751, 0.31121588, 0.99697732, 5.04261588, 0.92047609, 0.61495549, 17.52584946]

# convert to numpy arrays
A_prl = np.asarray(A_prl)
Q_prl = np.asarray(Q_prl)

# convert A_prl to Q_A
Q_Aprl = C * A_prl**(3/2)

# plot
plt.scatter(Q_Aprl,Q_prl,marker='.',color='tab:blue',s=2*s_val,label=r'$\textsc{gene}$')


# add reviewer's data 
Q_Rev = [23.46,12.26]

# make Miller inputs of reviewer's data
omn = 7.8 
eta = 0
eps = 0.2
q = 2.01
kappa =1.37
delta = 0.16
drR0 = -0.23
sq = 1.59
skappa = 0.20
sdelta = 0.33
alpha  =0.17
# calculate AE
AE_rev_1 = AEtok.calc_AE(omn,eta,eps,q,kappa,delta,drR0,sq,skappa,sdelta,alpha,L_ref='major')
AE_rev_2 = AEtok.calc_AE(omn,eta,eps,q,kappa,-delta,drR0,sq,skappa,sdelta,alpha,L_ref='major')
# convert to Q_A
Q_A_rev_1 = C * AE_rev_1**(3/2)
Q_A_rev_2 = C * AE_rev_2**(3/2)

# plot
plt.scatter(Q_A_rev_1,Q_Rev[0],marker='.',color='tab:orange',s=2*s_val,label=r'Rev. I')
plt.scatter(Q_A_rev_2,Q_Rev[1],marker='.',color='tab:orange',s=2*s_val)


# add legend
plt.legend()


x_arr = np.linspace(1e-1,1e4,100)
y_arr = x_arr
plt.plot(x_arr,y_arr,color='red',linestyle='--')
plt.xlabel(r'$\widehat{Q}_{A}$')
plt.ylabel(r'$\widehat{Q}_e$')
plt.xscale('log')
plt.yscale('log')
# save figure
print('N =',N + len(Q_Aprl))
plt.savefig('Qe_QAE_TGLF_scatter.png',dpi=1000)
plt.show()