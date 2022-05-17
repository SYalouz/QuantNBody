# ==========================================================================================
# ==========================================================================================
# 
# A code to demonstrate how to implement a State-Averaged CASSCF method with 
# the QuantNBody package on the formaldimine molecule.
# The results are compared to the SA-CASSCF method already implemented in the
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

#%%

def RUN_SACASSCF_PSI4( string_geo,
                       basisset,
                       N_MO_Optimized, 
                       active_indices,
                       frozen_indices,
                       virtual_indices ):
    '''
    A function to run a SA-CASSCF method with the Psi4 package
    '''
    restricted_UOCC =  N_MO_Optimized - ( len(frozen_indices) + len(active_indices) )
    
    psi4.geometry( string_geo )
    psi4.set_options({ 'basis': basisset,
                        'DETCI_FREEZE_CORE' : False,
                        'reference': 'RHF',
                        'scf_type': 'DIRECT', # set e_convergence and d_convergence to 1e-8 instead of 1e-6
                        'num_roots': 2,
                        'frozen_docc':[0],
                        'restricted_docc': [ len(frozen_indices) ],
                        'active': [ len(active_indices) ], 
                        'restricted_uocc': [ restricted_UOCC ],  
                        'frozen_uocc': [ 0 ],
                        'mcscf_maxiter': 2000,                       
                        'avg_states'      : [ 0, 1 ],
                        'avg_weights'     : [ 0.5, 0.5 ], 
                        'S' : 0,
                        # 'MCSCF_R_CONVERGENCE' : 1e-7,
                        # 'MCSCF_E_CONVERGENCE' : 1e-7,
                        'MCSCF_ALGORITHM' : 'AH'
                        })
    
    Escf, wfnSCF = psi4.energy('scf', return_wfn=True)
    sacasscf, sacasscf_wfn = psi4.energy('casscf',return_wfn=True)

    E0_sacasscf = psi4.variable('CI ROOT 0 TOTAL ENERGY')
    E1_sacasscf = psi4.variable('CI ROOT 1 TOTAL ENERGY')
    
    return E0_sacasscf, E1_sacasscf 

#%%

psi4.core.set_output_file("output_Psi4.txt", False)

#========================================================|
# Parameters for the simulation
# General Quantum chemistry parameters  =======
basisset         = 'sto-3g'
nelec_active     = 2   #   Number of active electrons in the Active-Space  
frozen_indices   = [ i for i in range(1) ]
active_indices   = [ i for i in range(1,5) ]
virtual_indices  = [ i for i in range(5,6) ] 
N_MO_total       = ( len(frozen_indices) 
                   + len(active_indices) 
                   + len(virtual_indices) ) # Total number of MOs

# Definition of the states weights : EQUI-ENSEMBLE 
w_A = w_B = 0.5

# Energy convergence criterium for the global optimization 
E_thresh        = 1e-4
OPT_OO_MAX_ITER = 25
Grad_threshold  = 1e-4

# Dimension of the many-body space 
dim_H  = math.comb( 2*len(active_indices), nelec_active ) 

# Building the Many-body basis            
nbody_basis = qnb.tools.build_nbody_basis( len(active_indices), nelec_active )     

# Building the matrix representation of the adagger_a operator in the many-body basis                       
a_dagger_a  = qnb.tools.build_operator_a_dagger_a( nbody_basis )   

# Building the matrix representation of several interesting spin operators in the many-body basis  
S_2, S_p, S_Z = qnb.tools.build_s2_sz_splus_operator( a_dagger_a ) 

#%%
  
list_r = np.linspace(0.25, 2.2, 10) 

E_0_qnb = []
E_1_qnb = []
for r in ( list_r ): 
    
    #========================================================|
    # Molecular geometry / Quantum chemistry calculations
    # Clean all previous options for psi4
    # psi4.core.clean()
    # psi4.core.clean_variables()
    # psi4.core.clean_options()
    
    # Li-H geometry  
    string_geo = """Li 0 0 0
                    H  0 0 {}
                    symmetry c1 """.format( r )
                    
    molecule = psi4.geometry(string_geo) 
    psi4.set_options({'basis'      : basisset,
                      'reference'  : 'rhf',
                      'SCF_TYPE'   : 'DIRECT' })
    
    scf_e, scf_wfn = psi4.energy( 'HF', molecule=molecule, return_wfn=True, verbose=0 )
    E_rep_nuc = molecule.nuclear_repulsion_energy()
    C_RHF     = np.asarray(scf_wfn.Ca()).copy()           # MO coeff matrix from the initial RHF calculation
    mints     = psi4.core.MintsHelper(scf_wfn.basisset()) # Get AOs integrals using MintsHelper
    Num_AO    = np.shape(np.asarray(mints.ao_kinetic()))[0]

    #%%
    # Construction of the first reference Hamiltonian / MO integrals
    C_ref = C_RHF # Initial MO coeff matrix  
    
    # Storing the 1/2-electron integrals in the original AO basis
    h_AO = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential()) 
    g_AO = np.asarray(mints.ao_eri()).reshape(( Num_AO, Num_AO, Num_AO, Num_AO )) 
     
    h_MO, g_MO  = qnb.tools.transform_1_2_body_tensors_in_new_basis( h_AO, g_AO, C_ref )
    
    E_core, h_, g_  = qnb.tools.qc_get_active_space_integrals(h_MO,
                                                              g_MO,
                                                              occupied_indices = frozen_indices,
                                                              active_indices   = active_indices)
   
    #%%
    # Building the matrix representation of the Hamiltonian operators 
    H  = qnb.tools.build_hamiltonian_quantum_chemistry(h_,
                                                       g_,
                                                       nbody_basis,
                                                       a_dagger_a,
                                                       S_2=S_2,
                                                       S_2_target=0,
                                                       penalty=100)
    
    # =======================================================|
    # STARTING GLOBAL OPTIMIZATION LOOP  ====================|
    # =======================================================|
    E_old      = 0
    E_new      = 1.e+99
    cycles_opt = 0
    
    while ( abs( E_new - E_old ) > E_thresh ):

        # Storing the energy for the very first iteration
        if ( cycles_opt > 0 ):
            E_old = E_new
    
        # Counter of optimization cycles
        cycles_opt += 1
    
        #========================================================|
        # EXACT DIAGONALIZER 
        # eig_en, eig_vec = scipy.sparse.linalg.eigsh( H, which='SA' ) 
        eig_en, eig_vec = scipy.linalg.eigh( H.A )     
        
        Psi_A = eig_vec[:,0]
        Psi_B = eig_vec[:,1]
        E_new = ( eig_en[0] + eig_en[1] ) /2.  + E_core + E_rep_nuc
        print("Energy :", E_new, eig_en[0]+ E_core +E_rep_nuc, eig_en[1]+ E_core + E_rep_nuc)
        #=======================================================|
        # Classical Orbital-Optimization process  
        print(" ====== Orb-Opt PHASE ====== ")
          
        # Building the SA One and Two-RDM from the wavefunctions
        one_rdm_A, two_rdm_A  = qnb.tools.build_full_mo_1rdm_and_2rdm( Psi_A,
                                                                      a_dagger_a,
                                                                      active_indices,
                                                                      N_MO_total )  
        one_rdm_B, two_rdm_B  = qnb.tools.build_full_mo_1rdm_and_2rdm( Psi_B,
                                                                      a_dagger_a,
                                                                      active_indices,
                                                                      N_MO_total ) 
        one_rdm_SA = w_A * one_rdm_A + w_B * one_rdm_B
        two_rdm_SA = w_A * two_rdm_A + w_B * two_rdm_B
            
        # Realizing the orbital-optimization process
        C_ref, E_new, h_MO, g_MO = qnb.tools.orbital_optimisation_newtonraphson(one_rdm_SA,
                                                                            two_rdm_SA,
                                                                            active_indices,
                                                                            frozen_indices,
                                                                            virtual_indices, 
                                                                            C_ref, 
                                                                            E_rep_nuc,
                                                                            h_AO,
                                                                            g_AO,
                                                                            N_MO_total,
                                                                            OPT_OO_MAX_ITER,
                                                                            Grad_threshold,
                                                                            TELL_ME=True)
        # Building the orbital-optimized Frozen-core Hamiltonian
        E_core, h_, g_  = qnb.tools.qc_get_active_space_integrals(h_MO,
                                                                  g_MO,
                                                                  occupied_indices=frozen_indices,
                                                                  active_indices=active_indices)
        
        # Building the matrix representation of the Hamiltonian operators 
        H  = qnb.tools.build_hamiltonian_quantum_chemistry( h_,
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
    # END OF GLOBAL OPTIMIZATION LOOP =======================|
    #========================================================|
#%%

# =======================================================|
# SIMILAR CALCULATION WITH THE PSI4 package =============|
#========================================================|
E_0_psi4 = []
E_1_psi4 = []
for r in tqdm( list_r ): 
    
    #========================================================|
    # Molecular geometry / Quantum chemistry calculations
    # Clean all previous options for psi4
    psi4.core.clean()
    psi4.core.clean_variables()
    psi4.core.clean_options()
     
    # Li-H geometry  
    string_geo = """Li 0 0 0
                    H  0 0 {}
                    symmetry c1 """.format( r )
                    
    E0_sacasscf, E1_sacasscf  =  RUN_SACASSCF_PSI4( string_geo,
                                                   basisset,
                                                   N_MO_total, 
                                                   active_indices,
                                                   frozen_indices,
                                                   virtual_indices )
    E_0_psi4 += [ E0_sacasscf ]
    E_1_psi4 += [ E1_sacasscf ]

#%%
plt.rc('font',  family='Helvetica')
plt.rc('mathtext', fontset='stix')
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('lines', linewidth='2')

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 6)) 
ax1.plot( list_r,  E_0_psi4, color='red',   label='$E_0^{psi4}$')
ax1.plot( list_r,  E_1_psi4, color='red',   label='$E_1^{psi4}$')
ax1.plot( list_r,  E_0_qnb, color='blue',  marker='o', label='$E_0^{qnb}$')
ax1.plot( list_r,  E_1_qnb, color='blue',  marker='o', label='$E_1^{qnb}$')
ax1.set_xlabel('Intertatomic distance $r_{Li-H}$ ($\AA$) ', size=22)
ax1.set_ylabel('Energy (Ha)', size=22)
ax1.autoscale(enable=True, axis='y', tight=None)
ax1.legend(fontsize='x-large')
ax1.grid()
plt.show()