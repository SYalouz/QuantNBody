# ==========================================================================================
# ==========================================================================================
# 
# A code to demonstrate how to implement a CASCI method with 
# the QuantNBody package on the H4 molecule.
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
sys.path.append('../')

import quantnbody as qnb

def RUN_CASCI_PSI4( string_geo,
                    basisset,
                    active_indices,
                    frozen_indices,
                    virtual_indices ):
    '''
    A function to run a CASCI method with the Psi4 package
    '''

    psi4.geometry( string_geo )
    psi4.set_options({ 'basis': basisset,
                        'reference': 'RHF',
                        'scf_type': 'DIRECT', # set e_convergence and d_convergence to 1e-8 instead of 1e-6
                        'num_roots': 2,
                        'frozen_docc' : [ len(frozen_indices) ],
                        'active'      : [ len(active_indices) ],
                        'frozen_uocc' : [ len(virtual_indices) ],
                        'S' : 0  })

    Escf, wfnSCF = psi4.energy('scf', return_wfn=True)
    casci, casci_wfn = psi4.energy('fci',return_wfn=True)

    E0_casci = psi4.variable('CI ROOT 0 TOTAL ENERGY')
    E1_casci = psi4.variable('CI ROOT 1 TOTAL ENERGY')

    return E0_casci, E1_casci

#%%

psi4.core.set_output_file("output_Psi4.txt", False)

#========================================================|
# Parameters for the simulation
# General Quantum chemistry parameters  =======
basisset         = 'sto-3g'
nelec_active     = 2   #   Number of active electrons in the Active-Space
frozen_indices   = [ i for i in range(1) ]
active_indices   = [ i for i in range(1,3) ]
virtual_indices  = [ i for i in range(3,4) ]

# Dimension of the many-body space 
dim_H  = math.comb( 2*len(active_indices) , nelec_active )

# Building the Many-body basis            
nbody_basis = qnb.fermionic.tools.build_nbody_basis( len(active_indices) , nelec_active )

# Building the matrix representation of the adagger_a operator in the many-body basis                       
a_dagger_a  = qnb.fermionic.tools.build_operator_a_dagger_a( nbody_basis )

# Building the matrix representation of several interesting spin operators in the many-body basis  
S_2, S_p, S_Z = qnb.fermionic.tools.build_s2_sz_splus_operator( a_dagger_a )

#%%

list_angle = np.linspace(0.15, 3., 18)

E_0_qnb = []
E_1_qnb = []

for angle in ( list_angle ):

    #========================================================|
    # Molecular geometry / Quantum chemistry calculations 
    # Li-H geometry  
    string_geo = qnb.fermionic.tools.generate_h4_geometry( 1., angle )

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

    h_MO, g_MO  = qnb.fermionic.tools.transform_1_2_body_tensors_in_new_basis( h_AO, g_AO, C_ref )

    E_core, h_, g_  = qnb.fermionic.tools.qc_get_active_space_integrals(h_MO,
                                                              g_MO,
                                                              frozen_indices = frozen_indices,
                                                              active_indices   = active_indices)
    #%%
    # Building the matrix representation of the Hamiltonian operators 
    H  = qnb.fermionic.tools.build_hamiltonian_quantum_chemistry(h_,
                                                       g_,
                                                       nbody_basis,
                                                       a_dagger_a,
                                                       S_2=S_2,
                                                       S_2_target=0,
                                                       penalty=100)

    eig_en, eig_vec = scipy.linalg.eigh( H.A )
    E_0_qnb += [ eig_en[0] + E_core + E_rep_nuc ]
    E_1_qnb += [ eig_en[1] + E_core + E_rep_nuc ]

# =======================================================|
# SIMILAR CALCULATION WITH THE PSI4 package =============|
#========================================================|
E_0_psi4 = [ ]
E_1_psi4 = [ ]
for angle in ( list_angle ):

    #========================================================|
    # Molecular geometry / Quantum chemistry calculations
    # Clean all previous options for psi4

    string_geo = qnb.fermionic.tools.generate_h4_geometry(1., angle)

    E0_casci, E1_casci  =  RUN_CASCI_PSI4( string_geo,
                                            basisset,
                                            active_indices,
                                            frozen_indices,
                                            virtual_indices  )

    E_0_psi4 += [ E0_casci ]
    E_1_psi4 += [ E1_casci ]

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
ax1.set_xlabel('H-H angle (rad.) ', size=22)
ax1.set_ylabel('Energy (Ha)', size=22)
ax1.autoscale(enable=True, axis='y', tight=None)
ax1.legend(fontsize='x-large')
ax1.grid()
plt.show()
