import scipy
import numpy as np
from itertools import combinations_with_replacement, combinations
from numba import njit, prange
import matplotlib.pyplot as plt
import scipy.sparse
import psi4


# =============================================================================
# CORE FUNCTIONS FOR THE BUILDING OF the "A_dagger A" OPERATOR
# ==========================================================================

def build_nbody_basis( n_mode, list_N_boson, n_mo, n_electron, S_z_cleaning=False ):
    """
    Create a many-body basis formed by a list of fock-state with a conserved total
    number of bosons

    Parameters
    ----------
    n_mode    : int
        Number of modes in total
    N_boson   :  int
        Number of bosons in total

    Returns
    -------
    nbody_basis : array
        List of many-body states (occupation number states)

    """
    # Building the N-electron many-body basis
    nbody_basis_fermions = []
    for combination in combinations(range(2 * n_mo), n_electron):
        fock_state_fermion = [0] * (2 * n_mo)
        for index in list(combination):
            fock_state_fermion[index] += 1
        nbody_basis_fermions += [fock_state_fermion]
    
    # In case we want to get rid of states with s_z != 0
    if S_z_cleaning:
        nbody_basis_fermion_cleaned = nbody_basis_fermions.copy()
        for i in range(np.shape(nbody_basis_fermions)[0]):
            s_z = check_sz(nbody_basis_fermions[i])
            if s_z != 0:
                nbody_basis_fermion_cleaned.remove(nbody_basis_fermions[i])
        nbody_basis_fermions = nbody_basis_fermion_cleaned
        
    nbody_basis_boson_and_fermion = []
    for n_boson in list_N_boson:
        
        for combination in combinations_with_replacement( range(n_mode), n_boson ):
            fock_state_boson = [ 0 for i in range(n_mode) ]
            for index in list(combination):
                fock_state_boson[ index ] += 1
                
            for fock_state_fermion in nbody_basis_fermions:
                nbody_basis_boson_and_fermion += [ fock_state_boson + fock_state_fermion ]
      
    # # In case we want only one electron, we simplify the states
    # if n_electron == 1:
    #     nbody_basis_cleaned = nbody_basis_boson_and_fermion.copy()
    #     for i in range(np.shape(nbody_basis_boson_and_fermion)[0]): 
    #         if i%2==1:
    #             nbody_basis_cleaned.remove(nbody_basis_boson_and_fermion[i])
    #     nbody_basis_boson_and_fermion = nbody_basis_cleaned
        
    return np.array( nbody_basis_boson_and_fermion )


def check_sz(ref_state):
    """
    Return the value of the S_z operator for a unique slater determinant
    (written as a list of occupation number)

    Parameters
    ----------
    ref_state :  vector
        Slater determinant (list of occupation numbers)

    Returns
    -------
    s_z_slater_determinant : float
        value of S_z for the given slater determinant

    Examples
    ________
    >>> check_sz([0, 0, 1, 0])  # in reference state there is one electron in second orbital with spin up (alpha).
    0.5

    >>> check_sz([1, 1, 0, 0])  # in reference state there is doubly occupied first MO
    0.0

    """
    s_z_slater_determinant = 0
    for elem in range(len(ref_state)):
        if elem % 2 == 0:
            s_z_slater_determinant += + 1 * ref_state[elem] / 2
        else:
            s_z_slater_determinant += - 1 * ref_state[elem] / 2

    return s_z_slater_determinant

@njit
def new_state_after_sq_boson_op(type_of_op, index_mode, ref_fock_state):
    """
    Parameters
    ----------
    type_of_op     :  type of bosonic operator to apply (creation of annihilation)
    index_mode     :  index of the second quantized mode to occupy/empty
    ref_fock_state :  initial state to be transformed

    Returns
    -------
    new_fock_state :  Resulting occupation number form of the transformed state
    coeff_phase    :  coefficient attached to the resulting state

    """
    new_fock_state    = ref_fock_state.copy()
    num_boson_in_mode = ref_fock_state[index_mode]
    coeff = 0
    if type_of_op == 'b':
        new_fock_state[index_mode] += -1
        coeff = np.sqrt( num_boson_in_mode )
    elif type_of_op == 'b^':
        new_fock_state[index_mode] += 1
        coeff = np.sqrt( num_boson_in_mode + 1 )

    return new_fock_state, coeff


# @njit
def boson_build_final_state_bd(ref_state, p, nbodybasis):
    state_one, coeff1 = new_state_after_sq_boson_op('b', p, ref_state) 
    kappa_ = nbodybasis.index( state_one.tolist() )
    return kappa_, coeff1



# numba -> njit version of build_operator_a_dagger_a
def build_boson_anihilation_operator_b( nbodybasis, n_mode, silent=True ):
    """
    Create a matrix representation of the a_dagger_a operator
    in the many-body basis

    Parameters
    ----------
    nbody_basis : array
        List of many-body states (occupation number states) (occupation number states)
    silent      : Boolean
        If it is True, function doesn't print anything when it generates a_dagger_a
    Returns
    -------
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator in the many-body basis

    """
    # Dimensions of problem
    dim_H  = len(nbodybasis) 
    # mapping_kappa = build_mapping(nbodybasis) # <== To be clearly improved

    b = np.zeros((n_mode), dtype=object) 
    for p in range( n_mode ): 
        b[p] = scipy.sparse.lil_matrix((dim_H, dim_H)) 

    for p in (range(n_mode)): 
        for kappa in range(dim_H):
            ref_state = nbodybasis[kappa]
            if ref_state[p] == 0:
                pass
            else :
                # kappa_, coeff1, coeff2 = build_final_state_ad_a(np.array(ref_state), p, q, mapping_kappa)
                kappa_, coeff = boson_build_final_state_bd(np.array(ref_state), p, nbodybasis.tolist()) # #  OLDER FUNCTION KEPT FOR DEBUGGING
                b[p][kappa_, kappa] = coeff 
    return b



# @njit
def fermion_build_final_state_ad_a(ref_state, p, q, nbodybasis, n_mode):
    """
    Create the final state generated after the consecutive application of the
    a_dagger_a operators on an initial state.

    Parameters
    ----------
    ref_state : Iterable[int]
        Initial stater to be modified
    p : int
        index of the mode where a fermion is created
    q : int
        index of the mode where a fermion is killed
    mapping_kappa : Iterable[int]
        Function creating the unique mapping between unique value of some configuration with its index in nbody_basis.

    Returns
    -------
    kappa_
        final index in the many-body basis for the resulting state
    p1 : int
        phase coefficient after removing electron
    p2 : int
        phase coefficient after adding new electron

    Examples
    ________
    >>> nbody_basis = build_nbody_basis(2, 2)  # 2 electrons and 2 MO
    >>> mapping = build_mapping(nbody_basis)
    >>> build_final_state_ad_a(np.array([1, 1, 0, 0]), 2, 1, mapping)  # exciting electron from spin MO 1 to spin MO 2
    (1, -1.0, -1.0)

    """ 
    state_one, p1   = new_state_after_sq_fermi_op('a' ,  q, ref_state )
    state_two, p2   = new_state_after_sq_fermi_op('a^',  p, state_one )  
    kappa_ = nbodybasis.index( state_two.tolist() )
    
    return kappa_, p1, p2 


# # @njit
# def fermion_build_final_state_ad_a(ref_state, p, q, nbodybasis, n_mode):
#     """
#     Create the final state generated after the consecutive application of the
#     a_dagger_a operators on an initial state.

#     Parameters
#     ----------
#     ref_state : Iterable[int]
#         Initial stater to be modified
#     p : int
#         index of the mode where a fermion is created
#     q : int
#         index of the mode where a fermion is killed
#     mapping_kappa : Iterable[int]
#         Function creating the unique mapping between unique value of some configuration with its index in nbody_basis.

#     Returns
#     -------
#     kappa_
#         final index in the many-body basis for the resulting state
#     p1 : int
#         phase coefficient after removing electron
#     p2 : int
#         phase coefficient after adding new electron

#     Examples
#     ________
#     >>> nbody_basis = build_nbody_basis(2, 2)  # 2 electrons and 2 MO
#     >>> mapping = build_mapping(nbody_basis)
#     >>> build_final_state_ad_a(np.array([1, 1, 0, 0]), 2, 1, mapping)  # exciting electron from spin MO 1 to spin MO 2
#     (1, -1.0, -1.0)

#     """
#     fermionic_state = ref_state[n_mode:].copy() 
#     state_one, p1   = new_state_after_sq_fermi_op('a', q, fermionic_state )
#     state_two, p2   = new_state_after_sq_fermi_op('a^', p, state_one)
#     full_state_two  = ref_state.tolist()[:n_mode]  + state_two.tolist() 
#     print()
#     print(ref_state)
#     # print(fermionic_state)
#     # print(state_two) 
#     print(full_state_two)
    
#     kappa_ = nbodybasis.index( full_state_two )
#     # kappa_ = 0
#     return kappa_, p1, p2 
     

# @njit
# def fermion_build_final_state_ad_a(ref_state, p, q, mapping_kappa, nbodybasis):
#     """
#     Create the final state generated after the consecutive application of the
#     a_dagger_a operators on an initial state.

#     Parameters
#     ----------
#     ref_state : Iterable[int]
#         Initial stater to be modified
#     p : int
#         index of the mode where a fermion is created
#     q : int
#         index of the mode where a fermion is killed
#     mapping_kappa : Iterable[int]
#         Function creating the unique mapping between unique value of some configuration with its index in nbody_basis.

#     Returns
#     -------
#     kappa_
#         final index in the many-body basis for the resulting state
#     p1 : int
#         phase coefficient after removing electron
#     p2 : int
#         phase coefficient after adding new electron

#     Examples
#     ________
#     >>> nbody_basis = build_nbody_basis(2, 2)  # 2 electrons and 2 MO
#     >>> mapping = build_mapping(nbody_basis)
#     >>> build_final_state_ad_a(np.array([1, 1, 0, 0]), 2, 1, mapping)  # exciting electron from spin MO 1 to spin MO 2
#     (1, -1.0, -1.0)

#     """
#     state_one, p1 = new_state_after_sq_fermi_op('a', q, ref_state)
#     state_two, p2 = new_state_after_sq_fermi_op('a^', p, state_one)
#     kappa_ = nbodybasis.index( state_two.tolist() )

#     return kappa_, p1, p2


@njit
def new_state_after_sq_fermi_op(type_of_op, index_mode, ref_fock_state):
    """
    Create the final state generated by the application of a second quantization
    creation/anihilation operator on an initial many-body state.

    Parameters
    ----------
    type_of_op : str
        Type of operator to apply ("a" for creation or "a^" for annihilation)
    index_mode : int
        index of the second quantized mode to occupy/empty
    ref_fock_state : np.array
        initial state to be transformed

    Returns
    -------
    new_fock_state : Iterable[int]
        Resulting occupation number form of the transformed state
    coeff_phase : int (1 or -1)
        Phase attached to the resulting state

    Raises
    ------
    Exception
        if type_of_op is not either "a" or "a^"

    Examples
    ________
    >>> qnb.new_state_after_sq_fermi_op("a", 0, np.array([1, 1, 0, 0]))  # annihilation of e- spin MO with index 0
    (array([0, 1, 0, 0]), 1.0)
    >>> qnb.new_state_after_sq_fermi_op("a^", 3, np.array([1, 1, 0, 0])) # creation of electron
    (array([1, 1, 0, 1]), 1.0)

    """
    new_fock_state = ref_fock_state.copy()
    coeff_phase = (-1.) ** np.sum(ref_fock_state[0:index_mode])
    if type_of_op == 'a':
        new_fock_state[index_mode] += -1
    elif type_of_op == 'a^':
        new_fock_state[index_mode] += 1
    else:
        raise Exception('type_of_op has to either be "a" or "a^"!')

    return new_fock_state, coeff_phase



# numba -> njit version of build_operator_a_dagger_a
def build_fermion_operator_a_dagger_a(nbody_basis, n_mode, silent=True):
    """
    Create a matrix representation of the a_dagger_a operator in the many-body basis
    
    Parameters
    ----------
    nbody_basis : array
        List of many-body states (occupation number states)
    silent : bool, default=True
        If it is True, function doesn't print anything when it generates a_dagger_a

    Returns
    -------
    a_dagger_a : array
        Matrix representation of the a_dagger_a operators

    Examples
    ________
    >>> nbody_basis = nbody_basis(2, 2)
    >>> a_dagger_a = build_operator_a_dagger_a(nbody_basis, True)
    >>> a_dagger_a[0,0] # Get access to the operator counting the electron in the first spinorbital

    """
    # Dimensions of problem
    dim_H = len(nbody_basis)  
    n_mo  = len(nbody_basis[0][n_mode:]) // 2
    n_elec = np.sum(nbody_basis[0][n_mode:])
    n_b_max = np.max(nbody_basis[:,:n_mode])
    a_dagger_a = np.zeros((2 * n_mo, 2 * n_mo), dtype=object)
    for p in range(2 * n_mo):
        for q in range(p, 2 * n_mo):
            a_dagger_a[p, q] = scipy.sparse.lil_matrix((dim_H, dim_H))
            a_dagger_a[q, p] = scipy.sparse.lil_matrix((dim_H, dim_H))
    
    test_nbody_basis_fermions = []
    for combination in combinations(range(2 * n_mo), n_elec):
        fock_state_fermion = [0] * (2 * n_mo)
        for index in list(combination):
            fock_state_fermion[index] += 1
        test_nbody_basis_fermions += [fock_state_fermion]
    test_nbody_basis_fermions = np.array( test_nbody_basis_fermions )    
    
    # include_broken_spin = False
    # if (  np.shape(test_nbody_basis_fermions) == np.shape(nbody_basis[:,n_mode:]) ):
    #     include_broken_spin = True 
    
    
    #temporary solution to the problem of mapping the hybrid basis; solve the S_z cleaning problem
    Full_spin_basis = True
    if  scipy.special.binom(2 * n_mo, n_elec)*scipy.special.binom(n_mode + n_b_max , n_b_max) == dim_H :
        Full_spin_basis = True
    else:
        Full_spin_basis = False
        print("You are actually using a truncated basis. Be careful, some mathematical properties,like the resolution of identity, may be incorrect.")
        
    
    # if (  np.shape(test_nbody_basis_fermions) == np.shape(nbody_basis[:,n_mode:]) ):
    
    # if (  np.shape(test_nbody_basis_fermions) == np.shape(nbody_basis[:test_nbody_basis_fermions.shape[0],n_mode:]) ):
    #     Full_spin_basis = False
        # print("DONC ? ", Full_spin_basis) 
    # print(test_nbody_basis_fermions)
    # print()
    # print(nbody_basis[:test_nbody_basis_fermions.shape[0],n_mode:])
    
    for MO_q in (range(n_mo)): 
        for MO_p in range(MO_q, n_mo): 
            for kappa in range(dim_H): 
                ref_state = nbody_basis[kappa]
                
                # Single excitation : spin alpha -- alpha
                p, q = 2 * MO_p  + n_mode, 2 * MO_q + n_mode
                if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
                    pass
                elif ref_state[q] == 1:
                    kappa_, p1, p2 = fermion_build_final_state_ad_a(np.array(ref_state), p, q, nbody_basis.tolist(), n_mode)
                    a_dagger_a[p-n_mode, q-n_mode][kappa_, kappa] = a_dagger_a[q-n_mode, p-n_mode][kappa, kappa_] = p1 * p2
                
                if n_elec > 1:
                    
                    # Single excitation : spin beta -- beta
                    p, q = 2 * MO_p + 1 + n_mode, 2 * MO_q + 1  + n_mode
                    if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
                        pass
                    elif ref_state[q] == 1:
                        kappa_, p1, p2 = fermion_build_final_state_ad_a(np.array(ref_state), p, q, nbody_basis.tolist(), n_mode)
                        a_dagger_a[p-n_mode, q-n_mode][kappa_, kappa] = a_dagger_a[q-n_mode, p-n_mode][kappa, kappa_] = p1 * p2
    
                    if Full_spin_basis and MO_p == MO_q:  # <=== Necessary to build the Spins operator but not really for Hamiltonians
                        
                        # Single excitation : spin beta -- alpha
                        p, q = 2 * MO_p + 1 + n_mode, 2 * MO_p + n_mode
                        if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
                            pass
                        elif ref_state[q] == 1:
                            kappa_, p1, p2 = fermion_build_final_state_ad_a(np.array(ref_state), p, q, nbody_basis.tolist(), n_mode)
                            a_dagger_a[p-n_mode, q-n_mode][kappa_, kappa] = a_dagger_a[q-n_mode, p-n_mode][kappa, kappa_] = p1 * p2
    
                            # Single excitation : spin alpha -- beta
                        p, q = 2 * MO_p + n_mode, 2 * MO_p + 1 + n_mode
    
                        if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
                            pass
                        elif ref_state[q] == 1:
                            kappa_, p1, p2 = fermion_build_final_state_ad_a(np.array(ref_state), p, q, nbody_basis.tolist(), n_mode)
                            a_dagger_a[p-n_mode, q-n_mode][kappa_, kappa] = a_dagger_a[q-n_mode, p-n_mode][kappa, kappa_] = p1 * p2
    if not silent:
        print()
        print('\t ===========================================')
        print('\t ====  The matrix form of a^a is built  ====')
        print('\t ===========================================')

    return a_dagger_a


# =============================================================================
#  MANY-BODY HAMILTONIANS (FERMI HUBBARD AND QUANTUM CHEMISTRY)
# =============================================================================


def build_hamiltonian_hubbard_holstein(h_fermion,
                                        U_fermion, 
                                        a_dagger_a,
                                        h_boson,
                                        b,
                                        Coupling_fermion_boson,
                                        nbody_basis,
                                        S_2=None,
                                        S_2_target=None,
                                        penalty=100,
                                        v_term=None,
                                        cut_off_integral=1e-8):
    """
    Create a matrix representation of the Fermi-Hubbard Hamiltonian in the
    many-body basis.

    Parameters
    ----------
    h_ : array
        One-body integrals
    U_ : array
        Two-body integrals
    nbody_basis :  array
        List of many-body states (occupation number states)
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator
    S_2 : array, default=None)
        Matrix representation of the S_2 operator (default is None)
    S_2_target : float, default=None
        Value of the S_2 mean value we want to target (default is None)
    penalty : float, default=100
        Value of the penalty term to penalize the states that do not respect the spin symmetry (default is 100).
    v_term : array, default=None
        dipolar interactions.

    Returns
    -------
    H_fermi_hubbard : array
        Matrix representation of the Fermi-Hubbard Hamiltonian in the many-body basis

    """
    # # Dimension of the problem
    dim_H = len(nbody_basis)
    n_mo = np.shape(h_fermion)[0]
    n_mode = np.shape(h_boson)[0]
     
    # global E_
    E_ = np.empty((n_mo, n_mo), dtype=object)
    for p in range(n_mo):
        for q in range(n_mo):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]
        
    # Building the N-electron Fermi-Hubbard matrix hamiltonian (Sparse)
    H_fermi_hubbard = scipy.sparse.csr_matrix((dim_H, dim_H))
    
    indices_one_electron_integrals = np.transpose((abs(h_fermion)>cut_off_integral).nonzero())
    for indices in indices_one_electron_integrals:
        p, q = indices  
        H_fermi_hubbard += E_[p, q] * h_fermion[p, q]
    
    indices_two_electron_integrals = np.transpose((abs(U_fermion)>cut_off_integral).nonzero())
    for indices in indices_two_electron_integrals:
        p, q, r, s = indices  
        H_fermi_hubbard += a_dagger_a[2 * p, 2 * q] @ a_dagger_a[2 * r + 1, 2 * s + 1] * U_fermion[p, q, r, s]
    
    if v_term is not None:
        indices_two_electron_integrals = np.transpose((abs(v_term)>cut_off_integral).nonzero())
        for indices in indices_two_electron_integrals:
            p, q, r, s = indices  
            H_fermi_hubbard +=  E_[p, q] @ E_[r, s] * v_term[p, q, r, s] 

    # Reminder : S_2 = S(S+1) and the total  spin multiplicity is 2S+1
    # with S = the number of unpaired electrons x 1/2
    # singlet    =>  S=0    and  S_2=0
    # doublet    =>  S=1/2  and  S_2=3/4
    # triplet    =>  S=1    and  S_2=2
    # quadruplet =>  S=3/2  and  S_2=15/4
    # quintet    =>  S=2    and  S_2=6
    if S_2 is not None and S_2_target is not None:
        s_2_minus_target = S_2 - S_2_target *  scipy.sparse.identity(dim_H)
        H_fermi_hubbard += s_2_minus_target @ s_2_minus_target * penalty
    
    H_boson = scipy.sparse.csr_matrix((dim_H, dim_H)) 
    H_fermion_boson = scipy.sparse.csr_matrix((dim_H, dim_H)) 
    for p in range(n_mode):
        H_fermion_boson += E_[p,p] @ ( b[p].T + b[p] ) * Coupling_fermion_boson[p] #* h_boson[p,p]
        # H_boson +=   h_boson[p,p] * ( b[p].T @ b[p]  + 0.5* scipy.sparse.identity(dim_H) )
        for q in range(n_mode): 
            # if p == q:
                H_boson +=  b[p].T @ b[q] * h_boson[p,q] 
            
    return H_fermi_hubbard + H_boson + H_fermion_boson



def build_hamiltonian_hubbard_QED(  h_fermion,
                                    U_fermion, 
                                    a_dagger_a, 
                                    omega_cav, 
                                    lambda_coupling, 
                                    dipole_integrals, 
                                    b, 
                                    nbody_basis,
                                    S_2=None,
                                    S_2_target=None,
                                    penalty=100,
                                    v_term=None,
                                    cut_off_integral=1e-8 ):
    """
    Create a matrix representation of the Fermi-Hubbard Hamiltonian in the
    many-body basis.

    Parameters
    ----------
    h_ : array
        One-body integrals
    U_ : array
        Two-body integrals
    nbody_basis :  array
        List of many-body states (occupation number states)
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator
    S_2 : array, default=None)
        Matrix representation of the S_2 operator (default is None)
    S_2_target : float, default=None
        Value of the S_2 mean value we want to target (default is None)
    penalty : float, default=100
        Value of the penalty term to penalize the states that do not respect the spin symmetry (default is 100).
    v_term : array, default=None
        dipolar interactions.

    Returns
    -------
    H_fermi_hubbard : array
        Matrix representation of the Fermi-Hubbard Hamiltonian in the many-body basis

    """
    # # Dimension of the problem
    dim_H = len(nbody_basis)
    n_mo = np.shape(h_fermion)[0] 
     
    # global E_
    E_ = np.empty((n_mo, n_mo), dtype=object)
    for p in range(n_mo):
        for q in range(n_mo):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]
        
    # Building the N-electron Fermi-Hubbard matrix hamiltonian (Sparse)
    H_fermi_hubbard = scipy.sparse.csr_matrix((dim_H, dim_H))
    
    indices_one_electron_integrals = np.transpose((abs(h_fermion)>cut_off_integral).nonzero())
    for indices in indices_one_electron_integrals:
        p, q = indices  
        H_fermi_hubbard += E_[p, q] * h_fermion[p, q]
    
    indices_two_electron_integrals = np.transpose((abs(U_fermion)>cut_off_integral).nonzero())
    for indices in indices_two_electron_integrals:
        p, q, r, s = indices  
        H_fermi_hubbard += a_dagger_a[2 * p, 2 * q] @ a_dagger_a[2 * r + 1, 2 * s + 1] * U_fermion[p, q, r, s]
    
    if v_term is not None:
        indices_two_electron_integrals = np.transpose((abs(v_term)>cut_off_integral).nonzero())
        for indices in indices_two_electron_integrals:
            p, q, r, s = indices  
            H_fermi_hubbard +=  E_[p, q] @ E_[r, s] * v_term[p, q, r, s] 

    # Reminder : S_2 = S(S+1) and the total  spin multiplicity is 2S+1
    # with S = the number of unpaired electrons x 1/2
    # singlet    =>  S=0    and  S_2=0
    # doublet    =>  S=1/2  and  S_2=3/4
    # triplet    =>  S=1    and  S_2=2
    # quadruplet =>  S=3/2  and  S_2=15/4
    # quintet    =>  S=2    and  S_2=6
    if S_2 is not None and S_2_target is not None:
        s_2_minus_target = S_2 - S_2_target *  scipy.sparse.identity(dim_H)
        H_fermi_hubbard += s_2_minus_target @ s_2_minus_target * penalty
    
    H_boson =  b[0].T @ b[0] * omega_cav
        
    dipole_op = scipy.sparse.csr_matrix((dim_H, dim_H)) 
    indices_one_electron_integrals = np.transpose((abs(dipole_integrals)>cut_off_integral).nonzero())
    for indices in indices_one_electron_integrals:
        p = indices[0]
        # print('TEST ', indices)
        dipole_op += dipole_integrals[p] * (a_dagger_a[2*p,2*p] + a_dagger_a[2*p+1,2*p+1]) 
 
    H_fermion_boson = ( lambda_coupling * omega_cav * dipole_op @ ( b[0].T + b[0] ) 
                      + lambda_coupling**2. * omega_cav * dipole_op@dipole_op       )
            
    return H_fermi_hubbard + H_boson + H_fermion_boson
     

def build_hamiltonian_pauli_fierz ( h_fermion,
                                    g_fermion, 
                                    dipole_vec_integrals,
                                    nuclear_dipole_XYZ,
                                    State_ref_mean_dipole,
                                    a_dagger_a,
                                    omega_cav,
                                    b,
                                    lambda_coupling,
                                    nbody_basis,
                                    E_,
                                    e_,
                                    S_2=None,
                                    S_2_target=None,
                                    penalty=100, 
                                    cut_off_integral=1e-8):
 
    # # Dimension of the problem
    dim_H       = len(nbody_basis)  
    H_chemistry = scipy.sparse.csr_matrix((dim_H, dim_H))
    indices_one_electron_integrals = np.transpose((abs(h_fermion)>cut_off_integral).nonzero())
    for indices in indices_one_electron_integrals:
        p, q = indices 
        H_chemistry += E_[p, q] * h_fermion[p, q]
    
    indices_two_electron_integrals = np.transpose((abs(g_fermion)>cut_off_integral).nonzero())
    for indices in indices_two_electron_integrals:
        p, q, r, s = indices 
        e_pqrs = E_[p, q] @ E_[r, s] 
        if q == r: 
            e_pqrs += - E_[p, s]
        H_chemistry += e_pqrs * g_fermion[p, q, r, s] / 2. 
    
    dipole_operator_X  = scipy.sparse.csr_matrix((dim_H, dim_H))
    dipole_operator_Y  = scipy.sparse.csr_matrix((dim_H, dim_H))
    dipole_operator_Z  = scipy.sparse.csr_matrix((dim_H, dim_H))
    indices_one_electron_integrals = np.transpose((abs(dipole_vec_integrals[0])>cut_off_integral).nonzero())
    for indices in indices_one_electron_integrals:
        p, q = indices
        dipole_operator_X  +=  E_[p, q] * dipole_vec_integrals[0][p, q]
    
    indices_one_electron_integrals = np.transpose((abs(dipole_vec_integrals[1])>cut_off_integral).nonzero())
    for indices in indices_one_electron_integrals:
        p, q = indices
        dipole_operator_Y  +=  E_[p, q] * dipole_vec_integrals[1][p, q]
        
    indices_one_electron_integrals = np.transpose((abs(dipole_vec_integrals[2])>cut_off_integral).nonzero())
    for indices in indices_one_electron_integrals:
        p, q = indices
        dipole_operator_Z  +=  E_[p, q] * dipole_vec_integrals[2][p, q]
    
    # Need to add the nuclear term to the sum over the electronic dipole integrals
    dipole_operator_X += nuclear_dipole_XYZ[0] *  scipy.sparse.identity(dim_H)
    dipole_operator_Y += nuclear_dipole_XYZ[1] *  scipy.sparse.identity(dim_H)
    dipole_operator_Z += nuclear_dipole_XYZ[2] *  scipy.sparse.identity(dim_H)
            
    mean_dipole_op = State_ref_mean_dipole.T @ ( dipole_operator_X + dipole_operator_Y  + dipole_operator_Z ) @ State_ref_mean_dipole
    fluc_dip_op_X =  dipole_operator_X - mean_dipole_op * scipy.sparse.identity(dim_H) 
    fluc_dip_op_Y =  dipole_operator_Y - mean_dipole_op *  scipy.sparse.identity(dim_H)
    fluc_dip_op_Z =  dipole_operator_Z - mean_dipole_op *  scipy.sparse.identity(dim_H)
    
    # Reminder : S_2 = S(S+1) and the total spin multiplicity is 2S+1
    # with S = the number of unpaired electrons x 1/2
    # singlet    =>  S=0    and  S_2=0
    # doublet    =>  S=1/2  and  S_2=3/4
    # triplet    =>  S=1    and  S_2=2
    # quadruplet =>  S=3/2  and  S_2=15/4
    # quintet    =>  S=2    and  S_2=6
    if S_2 is not None and S_2_target is not None:
        s_2_minus_target = S_2 - S_2_target *  scipy.sparse.identity(dim_H)
        H_chemistry += s_2_minus_target @ s_2_minus_target * penalty
    
    p = 0 
    H_boson = scipy.sparse.csr_matrix((dim_H, dim_H)) 
    H_boson =  b[p].T @ b[p] * omega_cav 
    
    H_fermion_boson = scipy.sparse.csr_matrix((dim_H, dim_H)) 
    H_fermion_boson += - np.sqrt(omega_cav/2.) * lambda_coupling[0] * fluc_dip_op_X @ ( b[p].T + b[p] )  
    H_fermion_boson += - np.sqrt(omega_cav/2.) * lambda_coupling[1] * fluc_dip_op_Y @ ( b[p].T + b[p] )  
    H_fermion_boson += - np.sqrt(omega_cav/2.) * lambda_coupling[2] * fluc_dip_op_Z @ ( b[p].T + b[p] )   
    
    H_fermion_boson += 0.5 * lambda_coupling[0]**2 * fluc_dip_op_X @ fluc_dip_op_X
    H_fermion_boson += 0.5 * lambda_coupling[1]**2 * fluc_dip_op_Y @ fluc_dip_op_Y
    H_fermion_boson += 0.5 * lambda_coupling[2]**2 * fluc_dip_op_Z @ fluc_dip_op_Z
    
    return H_chemistry + H_boson + H_fermion_boson




def new_build_hamiltonian_pauli_fierz ( h_fermion,
                                        g_fermion, 
                                        dipole_vec_integrals,
                                        mean_dipole_op,
                                        a_dagger_a,
                                        omega_cav,
                                        b,
                                        lambda_coupling,
                                        nbody_basis,
                                        S_2=None,
                                        S_2_target=None,
                                        penalty=100,
                                        v_term=None):
 
    # # Dimension of the problem
    dim_H = len(nbody_basis)
    n_mo = np.shape(h_fermion)[0] 
    
    E_ = np.empty((n_mo, n_mo), dtype=object)
    e_ = np.empty((n_mo, n_mo, n_mo, n_mo), dtype=object) 
     
    for p in range(n_mo):
        for q in range(n_mo):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]
      
    H_chemistry         = scipy.sparse.csr_matrix((dim_H, dim_H))
    dipole_operator_X  = scipy.sparse.csr_matrix((dim_H, dim_H))
    dipole_operator_Y  = scipy.sparse.csr_matrix((dim_H, dim_H))
    dipole_operator_Z  = scipy.sparse.csr_matrix((dim_H, dim_H))
    for p in range(n_mo):
        for q in range(n_mo):
            H_chemistry += E_[p, q] * h_fermion[p, q]
            dipole_operator_X  +=  E_[p, q] * dipole_vec_integrals[0][p, q]
            dipole_operator_Y  +=  E_[p, q] * dipole_vec_integrals[1][p, q]
            dipole_operator_Z  +=  E_[p, q] * dipole_vec_integrals[2][p, q]
                
            for r in range(n_mo):
                for s in range(n_mo):
                    e_[p, q, r, s] = E_[p, q] @ E_[r, s]
                    if q == r:
                        e_[p, q, r, s] += - E_[p, s] 
                    H_chemistry += e_[p, q, r, s] * g_fermion[p, q, r, s] / 2.
                    
    fluc_dip_op_X =  dipole_operator_X - mean_dipole_op * scipy.sparse.identity(dim_H) 
    fluc_dip_op_Y =  dipole_operator_Y - mean_dipole_op * scipy.sparse.identity(dim_H)
    fluc_dip_op_Z =  dipole_operator_Z - mean_dipole_op * scipy.sparse.identity(dim_H)
    
    # Reminder : S_2 = S(S+1) and the total spin multiplicity is 2S+1
    # with S = the number of unpaired electrons x 1/2
    # singlet    =>  S=0    and  S_2=0
    # doublet    =>  S=1/2  and  S_2=3/4
    # triplet    =>  S=1    and  S_2=2
    # quadruplet =>  S=3/2  and  S_2=15/4
    # quintet    =>  S=2    and  S_2=6
    if S_2 is not None and S_2_target is not None:
        s_2_minus_target = S_2 - S_2_target *  scipy.sparse.identity(dim_H)
        H_chemistry += s_2_minus_target @ s_2_minus_target * penalty
    
    p = 0
    
    H_boson = scipy.sparse.csr_matrix((dim_H, dim_H)) 
    H_boson =  b[p].T @ b[p] * omega_cav 
    
    H_fermion_boson = scipy.sparse.csr_matrix((dim_H, dim_H)) 
    H_fermion_boson += - np.sqrt(omega_cav/2.) * lambda_coupling[0] * fluc_dip_op_X @ ( b[p].T + b[p] )  
    H_fermion_boson += - np.sqrt(omega_cav/2.) * lambda_coupling[1] * fluc_dip_op_Y @ ( b[p].T + b[p] )  
    H_fermion_boson += - np.sqrt(omega_cav/2.) * lambda_coupling[2] * fluc_dip_op_Z @ ( b[p].T + b[p] )   
    
    Self_dip_term = (  lambda_coupling[0] * fluc_dip_op_X 
                     + lambda_coupling[1] * fluc_dip_op_Y 
                     + lambda_coupling[2] * fluc_dip_op_Z )
    
    H_fermion_boson += 0.5 * Self_dip_term @ Self_dip_term 
    
    return H_chemistry + H_boson + H_fermion_boson



def cqed_rhf(lambda_vector, molecule_string, psi4_options_dict):
    """Computes the QED-RHF energy and density

    Arguments
    ---------
    lambda_vector : 1 x 3 array of floats
        the electric field vector, see e.g. Eq. (1) in [DePrince:2021:094112]
        and (15) in [Haugland:2020:041043]

    molecule_string : string
        specifies the molecular geometry

    options_dict : dictionary
        specifies the psi4 options to be used in running the canonical RHF

    Returns
    -------
    cqed_rhf_dictionary : dictionary
        Contains important quantities from the cqed_rhf calculation, with keys including:
            'RHF ENERGY' -> result of canonical RHF calculation using psi4 defined by molecule_string and psi4_options_dict
            'CQED-RHF ENERGY' -> result of CQED-RHF calculation, see Eq. (13) of [McTague:2021:ChemRxiv]
            'CQED-RHF C' -> orbitals resulting from CQED-RHF calculation
            'CQED-RHF DENSITY MATRIX' -> density matrix resulting from CQED-RHF calculation
            'CQED-RHF EPS'  -> orbital energies from CQED-RHF calculation
            'PSI4 WFN' -> wavefunction object from psi4 canonical RHF calcluation
            'CQED-RHF DIPOLE MOMENT' -> total dipole moment from CQED-RHF calculation (1x3 numpy array)
            'NUCLEAR DIPOLE MOMENT' -> nuclear dipole moment (1x3 numpy array)
            'DIPOLE ENERGY' -> See Eq. (14) of [McTague:2021:ChemRxiv]
            'NUCLEAR REPULSION ENERGY' -> Total nuclear repulsion energy

    Example
    -------
    >>> cqed_rhf_dictionary = cqed_rhf([0., 0., 1e-2], '''\nMg\nH 1 1.7\nsymmetry c1\n1 1\n''', psi4_options_dictionary)

    """
    # define geometry using the molecule_string
    mol = psi4.geometry(molecule_string)
    # define options for the calculation
    psi4.set_options(psi4_options_dict)
    # run psi4 to get ordinary scf energy and wavefunction object
    psi4_rhf_energy, wfn = psi4.energy("scf", return_wfn=True)

    # Create instance of MintsHelper class
    mints = psi4.core.MintsHelper(wfn.basisset())

    # Grab data from wavfunction
    # number of doubly occupied orbitals
    ndocc = wfn.nalpha()

    # grab all transformation vectors and store to a numpy array
    C = np.asarray(wfn.Ca())

    # use canonical RHF orbitals for guess CQED-RHF orbitals
    Cocc = C[:, :ndocc]

    # form guess density
    D = np.einsum("pi,qi->pq", Cocc, Cocc)  # [Szabo:1996] Eqn. 3.145, pp. 139

    # Integrals required for CQED-RHF
    # Ordinary integrals first
    V = np.asarray(mints.ao_potential())
    T = np.asarray(mints.ao_kinetic())
    I = np.asarray(mints.ao_eri())

    # Extra terms for Pauli-Fierz Hamiltonian
    # nuclear dipole
    mu_nuc_x = mol.nuclear_dipole()[0]
    mu_nuc_y = mol.nuclear_dipole()[1]
    mu_nuc_z = mol.nuclear_dipole()[2]

    # electronic dipole integrals in AO basis
    mu_ao_x = np.asarray(mints.ao_dipole()[0])
    mu_ao_y = np.asarray(mints.ao_dipole()[1])
    mu_ao_z = np.asarray(mints.ao_dipole()[2])

    # \lambda \cdot \mu_el (see within the sum of line 3 of Eq. (9) in [McTague:2021:ChemRxiv])
    l_dot_mu_el =  lambda_vector[0] * mu_ao_x
    l_dot_mu_el += lambda_vector[1] * mu_ao_y
    l_dot_mu_el += lambda_vector[2] * mu_ao_z

    # compute electronic dipole expectation value with
    # canonincal RHF density
    mu_exp_x = np.einsum("pq,pq->", 2 * mu_ao_x, D)
    mu_exp_y = np.einsum("pq,pq->", 2 * mu_ao_y, D)
    mu_exp_z = np.einsum("pq,pq->", 2 * mu_ao_z, D)

    # need to add the nuclear term to the sum over the electronic dipole integrals
    mu_exp_x += mu_nuc_x
    mu_exp_y += mu_nuc_y
    mu_exp_z += mu_nuc_z

    rhf_dipole_moment = np.array([mu_exp_x, mu_exp_y, mu_exp_z])

    # We need to carry around the electric field dotted into the nuclear dipole moment
    # and the electric field dotted into the RHF electronic dipole expectation value
    # see prefactor to sum of Line 3 of Eq. (9) in [McTague:2021:ChemRxiv]

    # \lambda_vector \cdot \mu_{nuc}
    l_dot_mu_nuc = (  lambda_vector[0] * mu_nuc_x
                    + lambda_vector[1] * mu_nuc_y
                    + lambda_vector[2] * mu_nuc_z )
    
    # \lambda_vecto \cdot < \mu > where <\mu> contains electronic and nuclear contributions
    l_dot_mu_exp = (  lambda_vector[0] * mu_exp_x
                    + lambda_vector[1] * mu_exp_y
                    + lambda_vector[2] * mu_exp_z )

    # dipole energy, Eq. (14) in [McTague:2021:ChemRxiv]
    #  0.5 * (\lambda_vector \cdot \mu_{nuc})** 2
    #      - (\lambda_vector \cdot <\mu> ) ( \lambda_vector\cdot \mu_{nuc})
    # +0.5 * (\lambda_vector \cdot <\mu>) ** 2
    d_c = 0.5 * l_dot_mu_nuc ** 2 - l_dot_mu_nuc * l_dot_mu_exp + 0.5 * l_dot_mu_exp ** 2

    # quadrupole arrays
    Q_ao_xx = np.asarray(mints.ao_quadrupole()[0])
    Q_ao_xy = np.asarray(mints.ao_quadrupole()[1])
    Q_ao_xz = np.asarray(mints.ao_quadrupole()[2])
    Q_ao_yy = np.asarray(mints.ao_quadrupole()[3])
    Q_ao_yz = np.asarray(mints.ao_quadrupole()[4])
    Q_ao_zz = np.asarray(mints.ao_quadrupole()[5])

    # Pauli-Fierz 1-e quadrupole terms, Line 2 of Eq. (9) in [McTague:2021:ChemRxiv]
    Q_PF = -0.5 * lambda_vector[0] * lambda_vector[0] * Q_ao_xx
    Q_PF -= 0.5 * lambda_vector[1] * lambda_vector[1] * Q_ao_yy
    Q_PF -= 0.5 * lambda_vector[2] * lambda_vector[2] * Q_ao_zz

    # accounting for the fact that Q_ij = Q_ji
    # by weighting Q_ij x 2 which cancels factor of 1/2
    Q_PF -= lambda_vector[0] * lambda_vector[1] * Q_ao_xy
    Q_PF -= lambda_vector[0] * lambda_vector[2] * Q_ao_xz
    Q_PF -= lambda_vector[1] * lambda_vector[2] * Q_ao_yz

    # Pauli-Fierz 1-e dipole terms scaled by
    # (\lambda_vector \cdot \mu_{nuc} - \lambda_vector \cdot <\mu>)
    # Line 3 in full of Eq. (9) in [McTague:2021:ChemRxiv]
    d_PF = ( l_dot_mu_nuc - l_dot_mu_exp ) * l_dot_mu_el

    # ordinary H_core
    H_0 = T + V

    # Add Pauli-Fierz terms to H_core. Eq. (11) in [McTague:2021:ChemRxiv]
    H = H_0 + Q_PF + d_PF

    # Overlap for DIIS
    S = mints.ao_overlap()
    # Orthogonalizer A = S^(-1/2) using Psi4's matrix power.
    A = mints.ao_overlap()
    A.power(-0.5, 1.0e-16)
    A = np.asarray(A)

    print("\nStart SCF iterations:\n")  
    Enuc = mol.nuclear_repulsion_energy()
    Eold = 0.0
    E_1el_crhf = np.einsum("pq,pq->", H_0 + H_0, D)
    E_1el = np.einsum("pq,pq->", H + H, D)
    print("Canonical RHF One-electron energy = %4.16f" % E_1el_crhf)
    print("CQED-RHF One-electron energy      = %4.16f" % E_1el)
    print("Nuclear repulsion energy          = %4.16f" % Enuc)
    print("Dipole energy                     = %4.16f" % d_c)

    # Set convergence criteria from psi4_options_dict
    if "e_convergence" in psi4_options_dict:
        E_conv = psi4_options_dict["e_convergence"]
    else:
        E_conv = 1.0e-7
    if "d_convergence" in psi4_options_dict:
        D_conv = psi4_options_dict["d_convergence"]
    else:
        D_conv = 1.0e-5 

    # maxiter
    maxiter = 500
    for SCF_ITER in range(1, maxiter + 1):

        # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141
        J = np.einsum("pqrs,rs->pq", I, D)
        K = np.einsum("prqs,rs->pq", I, D)

        # Pauli-Fierz 2-e dipole-dipole terms, line 2 of Eq. (12) in [McTague:2021:ChemRxiv]
        M = np.einsum("pq,rs,rs->pq", l_dot_mu_el, l_dot_mu_el, D)
        N = np.einsum("pr,qs,rs->pq", l_dot_mu_el, l_dot_mu_el, D)

        # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141
        # plus Pauli-Fierz terms Eq. (12) in [McTague:2021:ChemRxiv]
        F = H + J * 2 - K + 2 * M - N

        diis_e = np.einsum("ij,jk,kl->il", F, D, S) - np.einsum("ij,jk,kl->il", S, D, F)
        diis_e = A.dot(diis_e).dot(A)
        dRMS   = np.mean(diis_e ** 2) ** 0.5

        # SCF energy and update: [Szabo:1996], Eqn. 3.184, pp. 150
        # Pauli-Fierz terms Eq. 13 of [McTague:2021:ChemRxiv]
        SCF_E = np.einsum("pq,pq->", F + H, D) + Enuc + d_c

        print( "SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E"
               % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS) )
        
        if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
            break

        Eold = SCF_E

        # Diagonalize Fock matrix: [Szabo:1996] pp. 145
        Fp = A.dot(F).dot(A)  # Eqn. 3.177
        e, C2 = np.linalg.eigh(Fp)  # Solving Eqn. 1.178
        C = A.dot(C2)  # Back transform, Eqn. 3.174
        Cocc = C[:, :ndocc]
        D = np.einsum("pi,qi->pq", Cocc, Cocc)  # [Szabo:1996] Eqn. 3.145, pp. 139

        # update electronic dipole expectation value
        mu_exp_x = np.einsum("pq,pq->", 2 * mu_ao_x, D)
        mu_exp_y = np.einsum("pq,pq->", 2 * mu_ao_y, D)
        mu_exp_z = np.einsum("pq,pq->", 2 * mu_ao_z, D)

        mu_exp_x += mu_nuc_x
        mu_exp_y += mu_nuc_y
        mu_exp_z += mu_nuc_z

        # update \lambda \cdot <\mu>
        l_dot_mu_exp = (  lambda_vector[0] * mu_exp_x
                        + lambda_vector[1] * mu_exp_y
                        + lambda_vector[2] * mu_exp_z )
        
        # Line 3 in full of Eq. (9) in [McTague:2021:ChemRxiv]
        d_PF = (l_dot_mu_nuc - l_dot_mu_exp) * l_dot_mu_el

        # update Core Hamiltonian
        H = H_0 + Q_PF + d_PF

        # update dipole energetic contribution, Eq. (14) in [McTague:2021:ChemRxiv]
        d_c = 0.5 * l_dot_mu_nuc ** 2 - l_dot_mu_nuc * l_dot_mu_exp + 0.5 * l_dot_mu_exp ** 2 

        if SCF_ITER == maxiter:
            psi4.core.clean()
            raise Exception("Maximum number of SCF cycles exceeded.")

    # print("Total time for SCF iterations: %.3f seconds \n" % (time.time() - t))
    print("QED-RHF   energy: %.8f hartree" % SCF_E)
    print("Psi4  SCF energy: %.8f hartree" % psi4_rhf_energy)

    rhf_one_e_cont = 2 * H_0   # note using H_0 which is just T + V, and does not include Q_PF and d_PF
    rhf_two_e_cont = J * 2 - K # note using just J and K that would contribute to ordinary RHF 2-electron energy
    pf_two_e_cont  = 2 * M - N

    SCF_E_One = np.einsum("pq,pq->", rhf_one_e_cont, D)
    SCF_E_Two = np.einsum("pq,pq->", rhf_two_e_cont, D)
    CQED_SCF_E_Two = np.einsum("pq,pq->", pf_two_e_cont, D)

    CQED_SCF_E_D_PF = np.einsum("pq,pq->", 2 * d_PF, D)
    CQED_SCF_E_Q_PF = np.einsum("pq,pq->", 2 * Q_PF, D)

    assert np.isclose(  SCF_E_One + SCF_E_Two + CQED_SCF_E_D_PF + CQED_SCF_E_Q_PF + CQED_SCF_E_Two,
                        SCF_E - d_c - Enuc )

    cqed_rhf_dict = {
        "RHF ENERGY": psi4_rhf_energy,
        "CQED-RHF ENERGY": SCF_E,
        "1E ENERGY": SCF_E_One,
        "2E ENERGY": SCF_E_Two,
        "1E DIPOLE ENERGY": CQED_SCF_E_D_PF,
        "1E QUADRUPOLE ENERGY": CQED_SCF_E_Q_PF,
        "2E DIPOLE ENERGY": CQED_SCF_E_Two,
        "CQED-RHF C": C,
        "CQED-RHF DENSITY MATRIX": D,
        "CQED-RHF EPS": e,
        "PSI4 WFN": wfn,
        "RHF DIPOLE MOMENT": rhf_dipole_moment,
        "CQED-RHF DIPOLE MOMENT": np.array([mu_exp_x, mu_exp_y, mu_exp_z]),
        "NUCLEAR DIPOLE MOMENT": np.array([mu_nuc_x, mu_nuc_y, mu_nuc_z]),
        "DIPOLE ENERGY": d_c,
        "NUCLEAR REPULSION ENERGY": Enuc,
    }

    return cqed_rhf_dict


# =============================================================================
#   SPIN OPERATORS
# =============================================================================

def build_s2_sz_splus_operator(a_dagger_a):
    """
    Create a matrix representation of the spin operators s_2, s_z and s_plus
    in the many-body basis.

    Parameters
    ----------
    a_dagger_a : array
        matrix representation of the a_dagger_a operator in the many-body basis.

    Returns
    -------
    s_2 : array
        matrix representation of the s_2 operator in the many-body basis.
    s_plus : array
        matrix representation of the s_plus operator in the many-body basis.
    s_z :  array
        matrix representation of the  s_z operator in the many-body basis.

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    dim_H = np.shape(a_dagger_a[0, 0].A)[0]
    s_plus = scipy.sparse.csr_matrix((dim_H, dim_H))
    s_z = scipy.sparse.csr_matrix((dim_H, dim_H))
    for p in range(n_mo): 
        s_plus += a_dagger_a[2 * p, 2 * p + 1]
        s_z += (a_dagger_a[2 * p, 2 * p] - a_dagger_a[2 * p + 1, 2 * p + 1]) / 2.

    s_2 = s_plus @ s_plus.T + s_z @ s_z - s_z

    return s_2, s_z, s_plus



# =============================================================================
#   FUNCTIONS TO CREAT PERSONALIZED MANY_BODY STATES AND PROJECTORS
# =============================================================================

def my_state( many_body_state, nbody_basis ):
    """
    Translate a many_body_state (occupation number list) into a many-body
    state referenced into a given Many-body basis.

    Parameters
    ----------
    many_body_state  : array
        occupation number list
    nbody_basis         : array
        List of many-body states (occupation number states)

    Returns
    -------
    state :  array
        The many body state translated into the "kappa" many-body basis

    """
    kappa = np.flatnonzero((nbody_basis == many_body_state).all(1))[0]  # nbody_basis.index(slater_determinant)
    state = np.zeros(np.shape(nbody_basis)[0])
    state[kappa] = 1.

    return state


# =============================================================================
#  FUNCTION TO HELP THE VISUALIZATION OF MANY-BODY WAVE FUNCTIONS
# =============================================================================

def visualize_wft(WFT, nbody_basis, n_mode, cutoff=0.005, atomic_orbitals=False,compact=False):
    """
    Print the decomposition of a given input wavefunction in a many-body basis.

    Parameters
    ----------
    WFT              : array
        Reference wave function
    nbody_basis      : array
        List of many-body states (occupation number states)
    cutoff           : array
        Cut off for the amplitudes retained (default is 0.005)
    atomic_orbitals  : Boolean
        If True then instead of 0/1 for spin orbitals we get 0/alpha/beta/2 for atomic orbitals

    Returns
    -------
        Terminal printing of the wavefunction

    """
    list_index = np.where(abs(WFT) > cutoff)[0]

    states = []
    coefficients = []
    for index in list_index:
        coefficients += [WFT[index]]
        states += [nbody_basis[index]]

    list_sorted_index = np.flip(np.argsort(np.abs(coefficients)))

    return_string = f'\n\t{"-" * 11}\n\t Coeff.     N-body state and index \n\t{"-" * 7}     {"-" * 22}\n'
    for index in list_sorted_index[0:15]:
        state = states[index]
        True_index_state =  np.flatnonzero((nbody_basis == state).all(1))[0]
        ket = "".join([str(elem) for elem in state[0:n_mode]]) + '⟩ ⊗ '

        if atomic_orbitals:
            ket += get_ket_in_atomic_orbitals(state[n_mode:], bra=False)
        else:
            if compact : 
                    
                # Build ket notation for spin orbitals
                ket += "|"
                ket_top = ""
                
                # Loop over pairs of elements (even index and odd index) and construct the ket
                for it, (even, odd) in enumerate(zip(state[n_mode::2], state[n_mode +1::2]), 1):
                    ket += str(it) * (even + odd)  # Add the orbital number based on occupation
                    ket_top += (" " if even == 1 else "") + ("_" if odd == 1 else "")  # Create upper part for spin
                
            else: 
                ket += '|' + "".join([str(elem) for elem in state[n_mode:]])
                

        # Format the output
        return_string += f'\t{" " *(len(f"{coefficients[index]:+1.5f}") + 5 + n_mode )}   {ket_top}     \n' if compact else ""
        return_string += f'\t{coefficients[index]:+1.5f}  |{ket}⟩    #{True_index_state} \n'  
    print( return_string )
    return return_string




def get_ket_in_atomic_orbitals(state, bra=False):

    ret_string = ""
    for i in range(len(state) // 2):
        if state[i * 2] == 1:
            if state[i * 2 + 1] == 1:
                ret_string += '2'
            else:
                ret_string += '\u03B1'
        elif state[i * 2 + 1] == 1:
            ret_string += '\u03B2'
        else:
            ret_string += '0'
    if bra:
        ret_string = '⟨' + ret_string + '|'
    else:
        ret_string = '|' + ret_string + '⟩'
    return ret_string


def bar_y(ax, coefficients, nbody_basis, size_label):
    """Plot horizontal bar charts for coefficients of a wavefunction."""
    # Create horizontal bars with colors based on the sign of the coefficients
    rects1 = ax.barh(range(len(coefficients)), np.abs(coefficients), color=[
        'red' if x < 0 else 'dodgerblue' for x in coefficients], edgecolor='black', linewidth=0.65, align='center')

    # Add labels to each bar with a bit of padding
    ax.bar_label(rects1, labels=[nbody_basis[j]for j in range(len(coefficients))],
                 fontsize=size_label, padding=2)

    # Add grid to the x-axis for better readability
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)

    return


# Main plotting function for wavefunctions
def plot_wavefunctions(WFT, nbody_basis, n_mode, list_states=[0], probability=False, cutoff=0.005,compact=False, label_props=["x-large", 16, 16, 16]):

    """
    Plot the wavefunctions choosen by the user.

    Parameters
    ----------
    WFT : numpy array
        A 2D array containing the wavefunctions.
    nbody_basis : list
        A list of the many-body basis states.
    list_states : list
        A list of the states to plot. Default is [0] the ground state
    n_mode : int
        The number of modes considered in the wavefunction
    probability : bool
        If True, plot the probabilities of the states. If False, plot the coefficients. Default is False
    cutoff : float
        The cutoff for the coefficients to be plotted. Default is 0.005
    label_props : list
        A list of argument that can be modified to fine-tune the size of the labels.
        the first element is the size of the bar labels, it can be modified to :
        "x-small", "small", "medium", "large", "x-large", "xx-large".
        The second,third and fourth elements are the size of the state label, the x-axis title and tick labels, it should be modified to :
        10, 12, 14, 16, 18, 20 
        if the user need it. Default is ["x-large",16,16,16]

    Returns
    -------
    None
    """
    N = len(list_states)  # Number of states to plot
    # Maximum coefficient for setting x-ticks
    max_coefficient = np.max(np.abs(WFT[:, list_states]))
    damping = 0.25  # Damping factor for x-ticks
    # Create a grid of subplots with shared x-axis

    # Create a grid of subplots with shared x-axis
    list_states = list_states[::-1]
    indices_by_state = [np.where(np.abs(WFT[:, i]) > cutoff)[
        0] for i in list_states]

    height = [len(indices) for indices in indices_by_state]
    gridspecs = {'height_ratios': height} if N > 1 else {'height_ratios': None}

    # Loop through each state to plot
    fig, axs = plt.subplots(N, 1, sharex=True, dpi=150, gridspec_kw=gridspecs)
    for i in range(N):
        state_index = list_states[i]
        ax = axs[i] if N > 1 else axs
        list_index = indices_by_state[i]

        # Get indices where the coefficients exceed the cutoff value
        list_index = np.where(np.abs(WFT[:, state_index]) > cutoff)[0]

        # Collect coefficients and corresponding many-body states
        coefficients = WFT[list_index,
                           state_index]**2 if probability else WFT[list_index, state_index]
        states = [nbody_basis[j] for j in list_index]

        # Sort coefficients and states by the absolute value of the coefficients
        combined = list(zip(np.abs(coefficients), coefficients, states))
        sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
        sorted_abs_coefficients, sorted_coefficients, sorted_nbody_basis = zip(
            *sorted_combined)

        # Set colors based on the sign of the coefficients (red for negative, blue for positive)
        L_sign = ["red" if x < 0 else "dodgerblue" for x in sorted_coefficients]

        # Format labels for the states
        L_label = []
        for state in sorted_nbody_basis:
            fermionic_part = state[n_mode:]
            bosonic_part = state[:n_mode]
            
            if compact: 
                comp_part = ""
                for it, (even, odd) in enumerate(zip(fermionic_part[::2], fermionic_part[1::2]),1):
                    comp_part += str(it)  if even == 1 else ""
                    comp_part += r"$\overline{" + str(it) + "}$" if odd == 1 else "" # Add the orbital number based on occupation
                fermionic_part = comp_part
            L_label.append(
                f'|{"".join(str(elem) for elem in bosonic_part)}⟩ ⊗ |{"".join(str(elem) for elem in fermionic_part)}⟩'
            )

        bar_y(ax, sorted_coefficients, L_label, label_props[0])
        # Remove y-ticks and plot the bars
        ax.set_yticks([])
        ax.text(-0.07, 0.5, rf'$|\Psi_{state_index}\rangle$', ha='center', va='center',
                transform=ax.transAxes, fontsize=label_props[1],   bbox=dict(
                    facecolor='lightgray',  # Light gray background
                    edgecolor='black',      # Black edge
                    boxstyle='round,pad=0.25',  # Rounded edges
                    alpha=0.75,
                    linewidth=1  # Edge thickness
                ))  # Centered text

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig.supxlabel('probability' if probability else 'Coefficient',
                  fontsize=label_props[2])
    # Set the x-ticks based on the maximum coefficient
    end_tick = max_coefficient**2 + damping if probability else max_coefficient + damping
    x_ticks = np.arange(0., end_tick, 0.1)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=label_props[3])
    # Show the plot
    plt.tight_layout()
    plt.show()

    return



# =============================================================================
# INTEGRAL TRANSFORMATIONS
# =============================================================================

def transform_1_2_body_tensors_in_new_basis(h_b1, g_b1, C):
    """
    Transform electronic integrals from an initial basis "B1" to a new basis "B2".
    The transformation is realized thanks to a passage matrix noted "C" linking
    both basis like
    
    .. math::
        
        | B2_l \\rangle =  \sum_p | B1_p \\rangle C_{pl} 

    with :math:`| B2_l \\rangle` and :math:`| B2_p \\rangle` are vectors of the
    basis B1 and B2 respectively.

    Parameters
    ----------
    h_b1 : array
        1-electron integral given in basis B1
    g_b1 : array
        2-electron integral given in basis B1
    C    : array
        Transfer matrix from the B1 to the B2 basis

    Returns
    -------
    h_b2 : array
        1-electron integral given in basis B2
    g_b2 : array
        2-electron integral given in basis B2

    """
    # Case of complex transformation
    if( np.iscomplexobj(C) ):
        h_b2 = np.einsum('pi,qj,pq->ij', C.conj(), C, h_b1, optimize=True)
        g_b2 = np.einsum('ap, bq, cr, ds, abcd -> pqrs', C.conj(), C.conj(), C, C, g_b1, optimize=True)
    
    # Case of real transformation
    else: 
        h_b2 = np.einsum('pi,qj,pq->ij', C, C, h_b1, optimize=True)
        g_b2 = np.einsum('ap, bq, cr, ds, abcd -> pqrs', C, C, C, C, g_b1, optimize=True)
    
    return h_b2, g_b2



def build_E_and_e_operators( a_dagger_a, n_mo ):
    """
    Build the spin-free "E" and "e" excitation many-body operators for quantum chemistry

    Parameters
    ----------
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator
    n_mo : int
        Number of molecular orbitals considered

    Returns
    -------
    E_ : array
        Spin-free "E" many-body operators

    e_ : array
        Spin-free "e" many-body operators

    Examples
    ________
    >>> nbody_basis = qnb.build_nbody_basis(2, 2)
    >>> a_dagger_a = qnb.build_operator_a_dagger_a(nbody_basis)
    >>> build_E_and_e_operators(a_dagger_a, 2)

    """
    E_ = np.empty((n_mo, n_mo), dtype=object)
    e_ = np.empty((n_mo, n_mo, n_mo, n_mo), dtype=object)

    for p in range(n_mo):
        for q in range(n_mo):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]

    for p in range(n_mo):
        for q in range(n_mo):
            for r in range(n_mo):
                for s in range(n_mo):
                    e_[p, q, r, s] = E_[p, q] @ E_[r, s]
                    if q == r:
                        e_[p, q, r, s] += - E_[p, s]
    return E_, e_



# =============================================================================
#  DIFFERENT TYPES OF REDUCED DENSITY-MATRICES
# =============================================================================

# BOSONIC =======

def build_bosonic_anihilation_rdm(WFT, b):
    """
    Create a hybrid alpha-beta 1 RDM for a given wave function
    (Note : alpha for the lines, and beta for the columns)

    Parameters
    ----------
    WFT        : array
        Wave function for which we want to build the 1-RDM
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    one_rdm_alpha_beta :  array
        spin-alpha-beta 1-RDM (alpha for the lines, and beta for the columns)

    """
    n_mode = np.shape(b)[0] 
    one_rdm = np.zeros((n_mode))
    for p in range(n_mode): 
        one_rdm[p] = WFT.T @ b[p] @ WFT

    return one_rdm



def build_bosonic_1rdm(WFT, b):
    """
    Create a hybrid alpha-beta 1 RDM for a given wave function
    (Note : alpha for the lines, and beta for the columns)

    Parameters
    ----------
    WFT        : array
        Wave function for which we want to build the 1-RDM
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    one_rdm_alpha_beta :  array
        spin-alpha-beta 1-RDM (alpha for the lines, and beta for the columns)

    """
    n_mode = np.shape(b)[0] 
    one_rdm = np.zeros((n_mode, n_mode))
    for p in range(n_mode):
        for q in range(n_mode):
            one_rdm[p, q] = WFT.T @ b[p].T @ b[q] @ WFT

    return one_rdm




# FERMIONIC =======
def build_fermionic_1rdm_alpha(WFT, a_dagger_a):
    """
    Create a spin-alpha 1 RDM for a given wave function

    Parameters
    ----------
    WFT : array
        Wave function for which we want to build the 1-RDM (expressed in the numerical basis)
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    one_rdm_alpha : array
        spin-alpha 1-RDM

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    one_rdm_alpha = np.zeros((n_mo, n_mo))
    for p in range(n_mo):
        for q in range(p, n_mo):
            one_rdm_alpha[p, q] = WFT.T @ a_dagger_a[2 * p, 2 * q] @ WFT
            one_rdm_alpha[q, p] = one_rdm_alpha[p, q]
    return one_rdm_alpha


def build_fermionic_1rdm_beta(WFT, a_dagger_a):
    """
    Create a spin-beta 1 RDM for a given wave function

    Parameters
    ----------
    WFT : array
        Wave function for which we want to build the 1-RDM
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    one_rdm_beta : array
        Spin-beta 1-RDM

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    one_rdm_beta = np.zeros((n_mo, n_mo))
    for p in range(n_mo):
        for q in range(p, n_mo):
            one_rdm_beta[p, q] = WFT.T @ a_dagger_a[2 * p + 1, 2 * q + 1] @ WFT
            one_rdm_beta[q, p] = one_rdm_beta[p, q]
    return one_rdm_beta


def build_fermionic_1rdm_spin_free(WFT, a_dagger_a):
    """
    Create a spin-free 1 RDM for a given wave function

    Parameters
    ----------
    WFT : array
        Wave function for which we want to build the 1-RDM
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    one_rdm : array
        Spin-free 1-RDM

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    one_rdm = np.zeros((n_mo, n_mo))
    for p in range(n_mo):
        for q in range(p, n_mo):
            E_pq = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]
            one_rdm[p, q] = WFT.T @ E_pq @ WFT
            one_rdm[q, p] = one_rdm[p, q]
    return one_rdm


def build_fermionic_2rdm_fh_on_site_repulsion(WFT, a_dagger_a, mask=None):
    """
    Create a 2-RDM for a given wave function following the structure of the Fermi Hubbard
    on-site repulsion operator (u[i,j,k,l] corresponds to a^+_i↑ a_j↑ a^+_k↓ a_l↓)

    Parameters
    ----------
    WFT : array
        Wave function for which we want to build the 2-RDM
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator
    mask :  array, default=None
        4D array is expected. Function is going to calculate only elements of 2rdm where mask is not 0.
        For default None the whole 2RDM is calculated.
        If we expect 2RDM to be very sparse (has only a few non-zero elements) then it is better to provide
        array that ensures that we won't calculate elements that are not going to be used in calculation of
        2-electron interactions.
    Returns
    -------
        two_rdm_fh : array
            2-RDM associated to the on-site-repulsion operator

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    two_rdm_fh = np.zeros((n_mo, n_mo, n_mo, n_mo))
    for p in range(n_mo):
        for q in range(n_mo):
            for r in range(n_mo):
                for s in range(n_mo):
                    if mask is None or mask[p, q, r, s] != 0:
                        two_rdm_fh[p, q, r, s] += WFT.T @ a_dagger_a[2 * p, 2 * q] @ a_dagger_a[2 * r + 1, 2 * s + 1] \
                                                  @ WFT
    return two_rdm_fh


def build_fermionic_2rdm_fh_dipolar_interactions(WFT, a_dagger_a, mask=None):
    """
    Create a spin-free 2 RDM for a given Fermi Hubbard wave function for dipolar interaction operator
    it corresponds to <psi|(a^+_i↑ a_j↑ + a^+_i↓ a_j↓)(a^+_k↑ a_l↑ + a^+_k↓ a_l↓)|psi>

    Parameters
    ----------
    WFT        :  array
        Wave function for which we want to build the 1-RDM
    a_dagger_a :  array
        Matrix representation of the a_dagger_a operator
    mask       :  array
        4D array is expected. Function is going to calculate only elements of 2rdm where mask is not 0.
        For default None the whole 2RDM is calculated.
        If we expect 2RDM to be very sparse (has only a few non-zero elements) then it is better to provide
        array that ensures that we won't calculate elements that are not going to be used in calculation of
        2-electron interactions.
    Returns
    -------
    two_rdm_fh : array
        2-RDM associated to the dipolar interaction operator

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    two_rdm_fh = np.zeros((n_mo, n_mo, n_mo, n_mo))
    big_E_ = np.empty((2 * n_mo, 2 * n_mo), dtype=object)
    for p in range(n_mo):
        for q in range(n_mo):
            big_E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]
    for p in range(n_mo):
        for q in range(n_mo):
            for r in range(n_mo):
                for s in range(n_mo):
                    if mask is None or mask[p, q, r, s] != 0:
                        two_rdm_fh[p, q, r, s] = WFT.T @ big_E_[p, q] @ big_E_[r, s] @ WFT
    return two_rdm_fh


def build_fermionic_2rdm_spin_free(WFT, a_dagger_a):
    """
    Create a spin-free 2 RDM for a given wave function

    Parameters
    ----------
    WFT        : array:
        Wave function for which we want to build the spin-free 2-RDM
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    two_rdm : array
        Spin-free 2-RDM

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    two_rdm = np.zeros((n_mo, n_mo, n_mo, n_mo))
    two_rdm[:] = np.nan
    global E_
    E_ = np.empty((2 * n_mo, 2 * n_mo), dtype=object)
    for p in range(n_mo):
        for q in range(n_mo):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]

    for p in range(n_mo):
        for q in range(p, n_mo):
            for r in range(p, n_mo):
                for s in range(p, n_mo):
                    if np.isnan(two_rdm[p, q, r, s]):
                        two_rdm[p, q, r, s] = WFT.T @ E_[p, q] @ E_[r, s] @ WFT
                        if q == r:
                            two_rdm[p, q, r, s] += - WFT.T @ E_[p, s] @ WFT

                        # Symmetry operations:
                        two_rdm[r, s, p, q] = two_rdm[p, q, r, s]
                        two_rdm[q, p, s, r] = two_rdm[p, q, r, s]
                        two_rdm[s, r, q, p] = two_rdm[p, q, r, s]

    return two_rdm


def build_fermionic_1rdm_and_2rdm_spin_free(WFT, a_dagger_a):
    """
    Create both spin-free 1- and 2-RDMs for a given wave function

    Parameters
    ----------
    WFT        : array
        Wave function for which we want to build the 1-RDM
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    one_rdm : array
        Spin-free 1-RDM
    two_rdm : array
        Spin-free 2-RDM

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    one_rdm = np.zeros((n_mo, n_mo))
    two_rdm = np.zeros((n_mo, n_mo, n_mo, n_mo))
    two_rdm[:] = np.nan
    global E_
    E_ = np.empty((2 * n_mo, 2 * n_mo), dtype=object)
    for p in range(n_mo):
        for q in range(n_mo):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]

    for p in range(n_mo):
        for q in range(p, n_mo):
            one_rdm[p, q] = WFT.T @ E_[p, q] @ WFT
            one_rdm[q, p] = one_rdm[p, q]
            for r in range(p, n_mo):
                for s in range(p, n_mo):
                    if np.isnan(two_rdm[p, q, r, s]):
                        two_rdm[p, q, r, s] = WFT.T @ E_[p, q] @ E_[r, s] @ WFT
                        if q == r:
                            two_rdm[p, q, r, s] += - WFT.T @ E_[p, s] @ WFT

                        # Symmetry operations:
                        two_rdm[r, s, p, q] = two_rdm[p, q, r, s]
                        two_rdm[q, p, s, r] = two_rdm[p, q, r, s]
                        two_rdm[s, r, q, p] = two_rdm[p, q, r, s]

    return one_rdm, two_rdm



def build_fermionic_hybrid_1rdm_alpha_beta(WFT, a_dagger_a):
    """
    Create a hybrid alpha-beta 1 RDM for a given wave function
    (Note : alpha for the lines, and beta for the columns)

    Parameters
    ----------
    WFT        : array
        Wave function for which we want to build the 1-RDM
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    one_rdm_alpha_beta :  array
        spin-alpha-beta 1-RDM (alpha for the lines, and beta for the columns)

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    one_rdm_alpha_beta = np.zeros((n_mo, n_mo))
    for p in range(n_mo):
        for q in range(n_mo):
            one_rdm_alpha_beta[p, q] = WFT.T @ a_dagger_a[2 * p, 2 * q + 1] @ WFT

    return one_rdm_alpha_beta



def build_fermionic_transition_1rdm_alpha(WFT_A, WFT_B, a_dagger_a):
    """
    Create a spin-alpha transition 1 RDM for a given wave function

    Parameters
    ----------
    WFT_A      :  array
        Left Wave function will be used for the Bra
    WFT_B      : array
        Right Wave function will be used for the Ket
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    transition_one_rdm_alpha : array
        transition spin-alpha 1-RDM

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    transition_one_rdm_alpha = np.zeros((n_mo, n_mo))
    for p in range(n_mo):
        for q in range(n_mo):
            transition_one_rdm_alpha[p, q] = WFT_A.T @ a_dagger_a[2 * p, 2 * q] @ WFT_B

    return transition_one_rdm_alpha



def build_fermionic_transition_1rdm_beta(WFT_A, WFT_B, a_dagger_a):
    """
    Create a spin-beta transition 1 RDM for a given wave function

    Parameters
    ----------
    WFT_A      : array
        Left Wave function will be used for the Bra
    WFT_B      : array
        Right Wave function will be used for the Ket
    a_dagger_a :  array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    transition_one_rdm_beta : array
        transition spin-beta 1-RDM

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    transition_one_rdm_beta = np.zeros((n_mo, n_mo))
    for p in range(n_mo):
        for q in range(n_mo):
            transition_one_rdm_beta[p, q] = WFT_A.T @ a_dagger_a[2 * p+1, 2 * q+1] @ WFT_B

    return transition_one_rdm_beta



def build_fermionic_transition_1rdm_spin_free(WFT_A, WFT_B, a_dagger_a):
    """
    Create a spin-free transition 1 RDM out of two given wave functions

    Parameters
    ----------
    WFT_A      :  array
        Left Wave function will be used for the Bra
    WFT_B      :  array
        Right Wave function will be used for the Ket
    a_dagger_a :  array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    transition_one_rdm : array
        spin-free transition 1-RDM

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    global E_
    E_ = np.empty((2 * n_mo, 2 * n_mo), dtype=object)
    for p in range(n_mo):
        for q in range(n_mo):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]

    transition_one_rdm = np.zeros((n_mo, n_mo))
    for p in range(n_mo):
        for q in range(n_mo):
            transition_one_rdm[p, q] = WFT_A.T @ E_[p,q] @ WFT_B

    return transition_one_rdm


def build_fermionic_transition_2rdm_spin_free(WFT_A, WFT_B, a_dagger_a):
    """
    Create a spin-free transition 2 RDM out of two given wave functions

    Parameters
    ----------
    WFT_A      :  array
        Left Wave function will be used for the Bra
    WFT_B      :  array
        Right Wave function will be used for the Ket
    a_dagger_a :  array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    transition_two_rdm : array
        Spin-free transition 2-RDM

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    global E_
    E_ = np.empty((2 * n_mo, 2 * n_mo), dtype=object)
    for p in range(n_mo):
        for q in range(n_mo):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]

    transition_two_rdm = np.zeros((n_mo, n_mo, n_mo, n_mo))
    for p in range(n_mo):
        for q in range(n_mo):
            for r in range(n_mo):
                for s in range(n_mo):
                    transition_two_rdm[p, q, r, s] = WFT_A.T @ E_[p, q] @ E_[r, s] @ WFT_B
                    if q == r:
                        transition_two_rdm[p, q, r, s] += - WFT_A.T @ E_[p, s] @ WFT_B

    return transition_two_rdm


