# ==========================================================================================
# ==========================================================================================
# 
# A code to demonstrate how to implement a FCI method with 
# the QuantNBody package on the Bose-Hubbard ring with 10 bosons in 4 sites.
#
# Author : Saad Yalouz
# ==========================================================================================
# ==========================================================================================

import numpy as np 
import psi4     
import math
import scipy 
import matplotlib.pyplot as plt
import sys 
sys.path.append('../')

import quantnbody as qnb
n_mode  = 4
n_boson = 10

nbodybasis = qnb.bosonic.tools.build_nbody_basis( n_mode, n_boson ) 
# mapping = qnb.bosonic.tools.build_mapping( nbodybasis ) 
dim_H = np.shape(nbodybasis)[0] 

#%% 
a_dagger_a = qnb.bosonic.tools.build_operator_a_dagger_a( nbodybasis )

 
#%%

h_ = np.zeros(( n_mode, n_mode )) 
for site in range(n_mode): 
    for site_ in range(n_mode):
        if (site != site_): 
            h_[site,site_] = h_[site_,site] = -1 

list_U = np.linspace(0, 2, 20) 

E_0_qnb = []
E_1_qnb = []
for U in  (list_U) : 
    U_  = np.zeros(( n_mode, n_mode, n_mode, n_mode )) 
    for site in range(n_mode):
        U_[ site, site, site, site ]  = -U/2
    
    # Building the matrix representation of the Hamiltonian operators 
    H = qnb.bosonic.tools.build_hamiltonian_bose_hubbard( h_,
                                                          U_,
                                                          nbodybasis,
                                                          a_dagger_a ) 
    eig_en, eig_vec = scipy.linalg.eigh( H.A  ) 
    E_0_qnb += [ eig_en[0] ]
    E_1_qnb += [ eig_en[1] ]

print()
print()
print(eig_en[:4])
 
#%%
plt.rc('font',  family='Helvetica') 
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('lines', linewidth='2')

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))  
ax1.plot( list_U,  E_0_qnb, color='blue',  marker='', label='$E_0^{qnb}$')
ax1.plot( list_U,  E_1_qnb, color='red',  marker='', label='$E_1^{qnb}$')
ax1.set_xlabel(' U ', size=22)
ax1.set_ylabel('Energy', size=22)
ax1.autoscale(enable=True, axis='y', tight=None)
ax1.legend(fontsize='x-large')
ax1.grid()
plt.show()