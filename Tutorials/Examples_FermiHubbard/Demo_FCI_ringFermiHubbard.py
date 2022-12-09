# ==========================================================================================
# ==========================================================================================
# 
# A code to demonstrate how to implement a FCI method with 
# the QuantNBody package on a Fermi-Hubbard ring with 4 electrons in 4 orbitals. 
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
 
#========================================================|
# Parameters for the simulation 
nelec_active =  6   #   Number of active electrons in the Active-Space  
n_mo         =  6   #   Number of molecular orbital

# Dimension of the many-body space 
dim_H  = math.comb( 2*n_mo, nelec_active ) 

# Building the Many-body basis            
nbody_basis = qnb.fermionic.tools.build_nbody_basis( n_mo, nelec_active )

# Building the matrix representation of the adagger_a operator in the many-body basis                       
a_dagger_a  = qnb.fermionic.tools.build_operator_a_dagger_a( nbody_basis )

# Building the matrix representation of several interesting spin operators in the many-body basis  
S_2, s_z, s_plus = qnb.fermionic.tools.build_s2_sz_splus_operator( a_dagger_a )

#%%
  
list_U = np.linspace(0, 6, 20)

# Hopping terms
h_MO = np.zeros((n_mo,n_mo))
for site in range(n_mo-1): 
    h_MO[site,site+1] =  h_MO[site+1,site] = -1
h_MO[0,n_mo-1] = h_MO[n_mo-1,0] =  -1

# MO energies 
# for site in range(n_mo): 
#     h_MO[site,site] =  - site
    
E_0_qnb = []
E_1_qnb = []
for U in  list_U : 
    U_MO = np.zeros((n_mo,n_mo,n_mo,n_mo))
    for site in range(n_mo):
        U_MO[site,site,site,site] = U
    
    # Building the matrix representation of the Hamiltonian operators 
    H  = qnb.fermionic.tools.build_hamiltonian_fermi_hubbard(h_MO,
                                                   U_MO,
                                                   nbody_basis,
                                                   a_dagger_a, 
                                                   S_2=S_2,
                                                   S_2_target=0,
                                                   penalty=100,
                                                   v_term=None )

    eig_en, eig_vec = scipy.linalg.eigh( H.A )
    E_0_qnb += [ eig_en[0]  ]
    E_1_qnb += [ eig_en[1]  ]
 
#%%
plt.rc('font',  family='Helvetica') 
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('lines', linewidth='2')

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))  
ax1.plot( list_U,  E_0_qnb, color='blue',  marker='o', label='$E_0^{qnb}$')
ax1.plot( list_U,  E_1_qnb, color='red',  marker='o', label='$E_1^{qnb}$')
ax1.set_xlabel(' U ', size=22)
ax1.set_ylabel('Energy', size=22)
ax1.autoscale(enable=True, axis='y', tight=None)
ax1.legend(fontsize='x-large')
ax1.grid()
plt.show()
