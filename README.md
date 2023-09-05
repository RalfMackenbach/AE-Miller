# AE-Miller
## A repo for calculating the available energy of trapped electrons in Miller tokamaks

### Dependencies
This code uses the BAD code ( https://github.com/RalfMackenbach/BAD ). It may readily be installed using 

```pip install -e.```

Please follow instructions on the page.

### Main idea
This code calculates the available energy (Æ) of any Miller tokamak (R. L. Miller _et al.;_ Noncircular, finite aspect ratio, local equilibrium model. PoP 1998; https://doi.org/10.1063/1.872666 ).

### Use
One can install the code by simply running 
```
pip install -e.
```
in the main directory.

There are two main functions of this code. 

#### s-α
If one wishes the calculate the Æ in the limit of a large-aspect-ratio tokamak with circular flux-surfaces one can use
```
calc_AE_salpha(omn,eta,epsilon,q,s_q,alpha,L_ref='major',A=3.0,rho=1.0,int_meth='quad',lam_res=1000,output='AE')
```
Documentation on the input parameters are provided as docstrings. 

#### Miller geometry
If one wishes to calculate the Æ of any Miller tokamak, one can use
```
calc_AE(omn,eta,epsilon,q,kappa,delta,dR0dr,s_q,s_kappa,s_delta,alpha,theta_res=1001,L_ref='major',A=3.0,rho=1.0,int_meth='quad',lam_res=1000,plot_precs=False,plot_dist=False,Cr_model='r_eff',output='AE')
```
Documentation on the input parameters are provided as docstrings. 


#### Important input parameters
Besides the plasma and geometric parameters, we highlight several important parameters.

`L_ref` sets which length-scale is used as reference length. For major, the major radius will be used and `omn = - R0/r dn/dr`. For minor, the minor radius will be used and `omn = - a/r dn/dr`. Furthermore, the expansion parameter rho* is adjusted for minor to `rho_* = rho_gyro / a`. One needs to explicitely set the aspect ratio `A` and the normalized flux-surface radius `rho = r / a` when running with `L_ref = 'minor'`.

`int_meth` sets the method by which the integral over pitch angle is performed. `quad` performs best in almost all cases, so it is recommended to run with this setting. However, if one wishes to investigate the Æ per bounce-well (`plot_precs=True` or `plot_dist=True`), one needs to run the code with `int_meth='trapz'`.

`Cr_model` sets the model for the correlation length. The r_eff model result in the best correlation with nonlinear heat-fluxes, so this is set as the standard. 

`output` can either be AE or Qe. The latter returns the heat-flux according to the scaling of PRL (2022 Mackenbach).

`theta_res` is the number of grid-points in theta used to generate the profiles, `lam_res` (only used if `int_meth='trapz'`) sets the resolution of the trapezoidal integral over pitch angle.
