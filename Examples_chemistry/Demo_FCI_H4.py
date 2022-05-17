# ==========================================================================================
# ==========================================================================================
# 
# A code to demonstrate how to implement a FCI method with 
# the QuantNBody package on the LiH molecule.
# The results are compared to the FCI method already implemented in the
# Psi4 quantum chemistry package.
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

def RUN_FCI_PSI4( string_geo,
                       basisset ):
    '''
    A function to run a FCI method with the Psi4 package
    ''' 
    psi4.core.clean()
    psi4.core.clean_variables()
    psi4.core.clean_options()
    
    psi4.geometry( string_geo )
    psi4.set_options({ 'basis': basisset, 
                      'num_roots': 2  })
    
    Escf, wfnSCF = psi4.energy('scf', return_wfn=True)
    fci, fci_wfn = psi4.energy('FCI',return_wfn=True)

    E0_fci = psi4.variable('CI ROOT 0 TOTAL ENERGY')
    E1_fci = psi4.variable('CI ROOT 1 TOTAL ENERGY')
    
    return E0_fci, E1_fci 

#%%

psi4.core.set_output_file("output_Psi4.txt", False)

#========================================================|
# Parameters for the simulation
# General Quantum chemistry parameters  =======
basisset     = 'sto-3g'
nelec_active = 4    
n_mo         = 4

# Dimension of the many-body space 
dim_H  = math.comb( 2*n_mo, nelec_active ) 

# Building the Many-body basis            
nbody_basis = qnb.tools.build_nbody_basis( n_mo, nelec_active )     

# Building the matrix representation of the adagger_a operator in the many-body basis                       
a_dagger_a  = qnb.tools.build_operator_a_dagger_a( nbody_basis )   

# Building the matrix representation of several interesting spin operators in the many-body basis  
S_2, S_p, S_Z = qnb.tools.build_s2_sz_splus_operator( a_dagger_a ) 

#%%
  
list_angle = np.linspace(0.25, 3., 18) 

E_0_qnb = []
E_1_qnb = []

for angle in tqdm( list_angle ): 
    
    #========================================================|
    # Molecular geometry / Quantum chemistry calculations 
    # Li-H geometry  
    string_geo =  qnb.tools_file.generate_h4_geometry( 1., angle )
                    
    molecule = psi4.geometry(string_geo) 
    psi4.set_options({'basis'      : basisset,
                      'reference'  : 'rhf',
                      'SCF_TYPE'   : 'DIRECT' })
    
    scf_e, scf_wfn = psi4.energy( 'HF', molecule=molecule, return_wfn=True, verbose=0 )
    E_rep_nuc = molecule.nuclear_repulsion_energy()
    C_RHF     = np.asarray(scf_wfn.Ca()).copy()             # MO coeff matrix from the initial RHF calculation
    mints     = psi4.core.MintsHelper(scf_wfn.basisset())   # Get AOs integrals using MintsHelper
    Num_AO    = np.shape(np.asarray(mints.ao_kinetic()))[0] 

    #%%
    # Construction of the first reference Hamiltonian / MO integrals
    C_ref = C_RHF # Initial MO coeff matrix  
    
    # Storing the 1/2-electron integrals in the original AO basis
    h_AO = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential()) 
    g_AO = np.asarray(mints.ao_eri()).reshape(( Num_AO, Num_AO, Num_AO, Num_AO )) 
     
    h_MO, g_MO  = qnb.tools.transform_1_2_body_tensors_in_new_basis( h_AO, g_AO, C_ref ) 
   
    #%%
    # Building the matrix representation of the Hamiltonian operators 
    H  = qnb.tools.build_hamiltonian_quantum_chemistry(h_MO,
                                                       g_MO,
                                                       nbody_basis,
                                                       a_dagger_a,
                                                       S_2=S_2,
                                                       S_2_target=0,
                                                       penalty=100)
    
    eig_en, eig_vec = scipy.linalg.eigh( H.A )
    E_0_qnb += [ eig_en[0]  + E_rep_nuc ]
    E_1_qnb += [ eig_en[1]  + E_rep_nuc ]
    
# =======================================================|
# SIMILAR CALCULATION WITH THE PSI4 package =============|
#========================================================|
E_0_psi4 = [ ]
E_1_psi4 = [ ]
for angle in tqdm( list_angle ):  
    
    #========================================================|
    # Molecular geometry / Quantum chemistry calculations
    # Clean all previous options for psi4
     
    string_geo = qnb.tools_file.generate_h4_geometry( 1, angle )
                    
    E0_fci, E1_fci  =  RUN_FCI_PSI4( string_geo,
                                     basisset  )

    E_0_psi4 += [ E0_fci ]
    E_1_psi4 += [ E1_fci ]

#%%
 
plt.rc('font',  family='Helvetica')
# plt.rc('mathtext', fontset='stix')
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('lines', linewidth='2')

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 6)) 
ax1.plot( list_angle,  E_0_psi4, color='red',  label='$E_0^{psi4}$')
ax1.plot( list_angle,  E_1_psi4, color='red',   label='$E_1^{psi4}$')
ax1.plot( list_angle,  E_0_qnb, color='blue', ls='', marker='o', label='$E_0^{qnb}$')
ax1.plot( list_angle,  E_1_qnb, color='blue', ls='', marker='o', label='$E_1^{qnb}$')
ax1.set_xlabel('Intertatomic distance $r_{Li-H}$ ($\AA$) ', size=22)
ax1.set_ylabel('Energy (Ha)', size=22)
ax1.autoscale(enable=True, axis='y', tight=None)
ax1.legend(fontsize='x-large')
ax1.grid()
plt.show()