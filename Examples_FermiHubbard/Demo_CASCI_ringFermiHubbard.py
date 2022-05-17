# ==========================================================================================
# ==========================================================================================
# 
# A code to demonstrate how to implement a CASCI method with 
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
from tqdm import tqdm
sys.path.append('../')

import quantnbody as qnb
 
#========================================================|
# Parameters for the simulation  
nelec_active     = 4   #   Number of active electrons in the Active-Space  
frozen_indices   = [ i for i in range(0) ]
active_indices   = [ i for i in range(0,3) ]
virtual_indices  = [ i for i in range(4,4) ]  

n_mo = len(frozen_indices) + len(active_indices) + len(virtual_indices)

# Dimension of the many-body space 
dim_H  = math.comb( 2*len(active_indices), nelec_active ) 

# Building the Many-body basis            
nbody_basis = qnb.tools.build_nbody_basis( len(active_indices), nelec_active )     

# Building the matrix representation of the adagger_a operator in the many-body basis                       
a_dagger_a  = qnb.tools.build_operator_a_dagger_a( nbody_basis )   

# Building the matrix representation of several interesting spin operators in the many-body basis  
S_2, s_z, s_plus = qnb.tools.build_s2_sz_splus_operator( a_dagger_a ) 

#%%

list_U = np.linspace(0, 6, 10)
  
# Hopping terms
h_MO = np.zeros(( n_mo, n_mo ))
for site in range(n_mo):
    for site_ in range(site,n_mo-1):
        h_MO[site,site_] =  h_MO[site_,site] = -1
h_MO[0,n_mo-1] = h_MO[n_mo-1,0] =  -1

# MO energies 
for site in range(n_mo): 
    h_MO[site,site] = - site
 
E_0_qnb = []
E_1_qnb = []
for U in  list_U : 
    U_MO = np.zeros((n_mo,n_mo,n_mo,n_mo))
    for site in range(n_mo):
        U_MO[site,site,site,site] = U
    
    # Preparing the active space integrals and the associated core contribution
    E_core, h_, U_  = qnb.tools.fh_get_active_space_integrals(h_MO,
                                                              U_MO,
                                                              frozen_indices=frozen_indices,
                                                              active_indices=active_indices )
 
    
    # Building the matrix representation of the Hamiltonian operators 
    H  = qnb.tools.build_hamiltonian_fermi_hubbard(h_, 
                                                   U_,
                                                   nbody_basis,
                                                   a_dagger_a, 
                                                   S_2=S_2,
                                                   S_2_target=0,
                                                   penalty=100,
                                                   v_term=None )

    eig_en, eig_vec = scipy.linalg.eigh( H.A )
    E_0_qnb += [ eig_en[0] + E_core ]
    E_1_qnb += [ eig_en[1] + E_core ]
 
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