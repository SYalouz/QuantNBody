# ==========================================================================================
# ==========================================================================================
#  .d88888b.                             888    888b    888 888888b.                 888          
# d88P" "Y88b                            888    8888b   888 888  "88b                888          
# 888     888                            888    88888b  888 888  .88P                888          
# 888     888 888  888  8888b.  88888b.  888888 888Y88b 888 8888888K.   .d88b.   .d88888 888  888 
# 888     888 888  888     "88b 888 "88b 888    888 Y88b888 888  "Y88b d88""88b d88" 888 888  888 
# 888 Y8b 888 888  888 .d888888 888  888 888    888  Y88888 888    888 888  888 888  888 888  888 
# Y88b.Y8b88P Y88b 888 888  888 888  888 Y88b.  888   Y8888 888   d88P Y88..88P Y88b 888 Y88b 888 
#  "Y888888"   "Y88888 "Y888888 888  888  "Y888 888    Y888 8888888P"   "Y88P"   "Y88888  "Y88888 
#        Y8b                                                                                  888 
# ====================================================================================== Y8b d88P 
#======================================================================================== "Y88P"  
#
# A sample code to demonstrate how to implement a FCI method with 
# the QuantNBody package on the Bose-Hubbard ring with 10 bosons on 4 sites.
#
# Author : Saad Yalouz 
# ==========================================================================================
# ==========================================================================================

import quantnbody as qnb
import numpy as np  
import scipy 
import matplotlib.pyplot as plt  

# Number of sites and bosons
num_sites = 4
num_bosons = 10

# Build the n-body basis
n_body_basis = qnb.bosonic.tools.build_nbody_basis(num_sites, num_bosons)  
dim_H = np.shape(n_body_basis)[0] 

# Build the creation and annihilation operators
a_dagger_a = qnb.bosonic.tools.build_operator_a_dagger_a(n_body_basis)
 

# Initialize the Hamiltonian matrix
h_ = np.zeros((num_sites, num_sites)) 
for site in range(num_sites): 
    for site_ in range(num_sites):
        if (site != site_): 
            h_[site, site_] = -1
           
# Generate a range of interaction strengths
list_U = np.linspace(0, 0.5, 20) 

# Lists to store ground and first excited state energies
ground_state_energies = []
first_excited_energies = []

# Iterate over each interaction strength
for U in list_U:
    # Build the interaction matrix
    U_ = np.zeros((num_sites, num_sites, num_sites, num_sites)) 
    for site in range(num_sites):
        U_[site, site, site, site] = -U
    
    # Build the Hamiltonian matrix
    H = qnb.bosonic.tools.build_hamiltonian_bose_hubbard(h_,
                                                          U_,
                                                          n_body_basis,
                                                          a_dagger_a) 
    
    # Diagonalize the Hamiltonian
    eig_en, eig_vec = scipy.linalg.eigh(H.A) 
    
    # Append the ground and first excited state energies
    ground_state_energies.append(eig_en[0])
    first_excited_energies.append(eig_en[1])

# Plot the energies
plt.rc('font', family='Helvetica') 
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('lines', linewidth='2')

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))  
ax1.plot(list_U, ground_state_energies, color='blue', marker='o', label='$E_0^{qnb}$')
ax1.plot(list_U, first_excited_energies, color='red', marker='o', label='$E_1^{qnb}$')
ax1.set_xlabel('U', size=22)
ax1.set_ylabel('Energy', size=22)
ax1.autoscale(enable=True, axis='y', tight=None)
ax1.legend(fontsize='x-large')
ax1.grid()
plt.show()
