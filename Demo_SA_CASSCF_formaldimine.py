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
import quantnbody as qnb
from tqdm import tqdm
import math
import scipy 

def RUN_SACASSCF_PSI4( string_geo,
                       basisset,
                       N_MO_Optimized, 
                       active_indices,
                       frozen_indices,
                       virtual_indices ):
    '''
    A function to run a SA-CASSCF method with the Psi4 package
    '''
    
    frozen_UOCC     = virtual_indices[-1] - N_MO_Optimized
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
                        'mcscf_maxiter': 1000,                       
                        'avg_states'      : [ 0, 1 ],
                        'avg_weights'     : [ 0.5, 0.5 ], 
                        'S' : 0 })
    
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
basisset         = 'cc-pvdz'
nelec_active     = 4   #   Number of active electrons in the Active-Space  
frozen_indices   = [ i for i in range(6) ]
active_indices   = [ i for i in range(6,10) ]
virtual_indices  = [ i for i in range(10,43) ] 
N_MO_total       = ( len(frozen_indices) 
                   + len(active_indices) 
                   + len(virtual_indices) ) # Total number of MOs


# Definition of the states weights : EQUI-ENSEMBLE 
w_A = w_B = 0.5

# Energy convergence criterium for the global optimization ( VQE + OO ) ====
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
 
phi        = 10. 
alpha_list = [ 130 ] 

 
for alpha in ( alpha_list ): 
    
    #========================================================|
    # Molecular geometry / Quantum chemistry calculations
    # Clean all previous options for psi4
    # psi4.core.clean()
    # psi4.core.clean_variables()
    # psi4.core.clean_options()
    
    # Formaldimine geometry 
    variables  = [ 1.498047, 1.066797, 0.987109, 118.359375 ] + [ alpha, phi ]
    string_geo = """0 1
                    N
                    C 1 {0}
                    H 2 {1}  1 {3}
                    H 2 {1}  1 {3} 3 180
                    H 1 {2}  2 {4} 3 {5}
                    symmetry c1 """.format( *variables )
                    
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
        eig_en, eig_vec = scipy.sparse.linalg.eigsh( H, which='SA' ) 
            
        
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
 
    # =======================================================|
    # END OF GLOBAL OPTIMIZATION LOOP =======================|
    #========================================================|
#%%

# =======================================================|
# SIMILAR CALCULATION WITH THE PSI4 package =============|
#========================================================|
for alpha in ( alpha_list ): 
    
    #========================================================|
    # Molecular geometry / Quantum chemistry calculations
    # Clean all previous options for psi4
    psi4.core.clean()
    psi4.core.clean_variables()
    psi4.core.clean_options()
     

    E0_sacasscf, E1_sacasscf  =  RUN_SACASSCF_PSI4( string_geo,
                                                   basisset,
                                                   43, 
                                                   active_indices,
                                                   frozen_indices,
                                                   virtual_indices )


    print( (E0_sacasscf+ E1_sacasscf)/2, E0_sacasscf, E1_sacasscf )


#%%
 
print()
print()
eig_en, eig_vec = scipy.linalg.eigh( H.A )

print("FINAL RESULTS :")
print("Energies obtained with our homemade SA-CASSCF code :")
print("E_SA :",  (eig_en[0] + eig_en[1])/2 + E_core + E_rep_nuc )
print("E_0  :",  eig_en[0] + E_core + E_rep_nuc )
print("E_1  :",   eig_en[1]+ E_core + E_rep_nuc )  
print("Energies obtained the SA-CASSCF method from Psi4   :")
print("E_SA :", (E0_sacasscf + E1_sacasscf)/2 )
print("E_0  :",  E0_sacasscf   )
print("E_1  :",  E1_sacasscf ) 