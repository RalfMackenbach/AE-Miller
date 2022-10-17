import random
import f90nml
import matplotlib.pyplot as plt
import sys
import numpy as np
random.seed(10)

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/ralfmackenbach/Documents/GitHub/AE-tok/Miller/scripts')
import AE_tokamak_calculation as AEtok





prec = 3 # "precision", i.e. number of decimal places

nml = f90nml.read('namelist.txt')




AE_list = []

idx=0

while idx<10:
    eta         = 0.0
    epsilon     = round(random.uniform(+0.1,  0.5),   prec)
    q           = round(random.uniform(+1.1,  4.0),   prec)
    kappa       = round(random.uniform(+1.0,  2.0),   prec)
    delta       = round(random.uniform(-1.0,  1.0),   prec)*epsilon
    dR0dr       = round(random.uniform(+0.0,  0.0),   prec)
    s_q         = round(random.uniform(-0.5,  0.5),   prec)
    s_kappa     = round(random.uniform(+0.0,  0.1),   prec)
    s_delta     = delta/np.sqrt((1-delta**2.0))
    alpha       = round(random.uniform(0.0,  0.2),   prec)
    omn         = round(random.uniform(3, 4),        prec)

    nml['geometry']['q0']       =   q
    nml['geometry']['shat']     =   s_q
    nml['geometry']['amhd']     =   alpha
    nml['geometry']['major_r']  =   1/epsilon # minor r set to unity
    nml['geometry']['trpeps']   =   epsilon
    nml['geometry']['kappa']    =   kappa
    nml['geometry']['delta']    =   delta
    nml['geometry']['zeta']     =   0.0
    nml['geometry']['s_kappa']  =   s_kappa
    nml['geometry']['s_delta']  =   s_delta
    nml['geometry']['s_zeta']   =   0.0
    nml['geometry']['drR']      =   dR0dr
    nml['species'][0]['omn']    =   omn
    nml['species'][1]['omn']    =   omn
    nml['in_out']['diagdir']    =   '/ptmp/ralfm/GENE_sims/miller/miller_{}/'.format(idx)


    AE_val = AEtok.calc_AE(omn,eta,epsilon,q,kappa,delta,dR0dr,s_q,s_kappa,s_delta,alpha,1001,1001,0.0,L_ref='minor',plot=False)
    print(AE_val)
    nml.write('parameters_miller{}.nml'.format(idx),force=True)
    idx = idx + 1
    AE_list.append(AE_val)

Q_avai = np.asarray(AE_list)
Q_GENE = np.asarray([24.74,4.72,14.68,44.08,7.79,133.21,631.37,70.97,41.15,15.57])
plt.scatter(np.log(Q_avai),np.log(Q_GENE))
plt.show()