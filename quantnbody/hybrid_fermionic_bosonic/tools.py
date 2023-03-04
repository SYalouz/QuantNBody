import scipy
import numpy as np
from itertools import combinations_with_replacement, combinations
from numba import njit, prange
import scipy.sparse

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
    
   
        
    # In case we want onely one electron, we simplify the states
    if n_electron == 1:
        nbody_basis_cleaned = nbody_basis_boson_and_fermion.copy()
        for i in range(np.shape(nbody_basis_boson_and_fermion)[0]): 
            if i%2==1:
                nbody_basis_cleaned.remove(nbody_basis_boson_and_fermion[i])
        nbody_basis_boson_and_fermion = nbody_basis_cleaned
        
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
    a_dagger_a = np.zeros((2 * n_mo, 2 * n_mo), dtype=object)
    for p in range(2 * n_mo):
        for q in range(p, 2 * n_mo):
            a_dagger_a[p, q] = scipy.sparse.lil_matrix((dim_H, dim_H))
            a_dagger_a[q, p] = scipy.sparse.lil_matrix((dim_H, dim_H))

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
                
                if n_elec != 1:
                    
                    # Single excitation : spin beta -- beta
                    p, q = 2 * MO_p + 1 + n_mode, 2 * MO_q + 1  + n_mode
                    if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
                        pass
                    elif ref_state[q] == 1:
                        kappa_, p1, p2 = fermion_build_final_state_ad_a(np.array(ref_state), p, q, nbody_basis.tolist(), n_mode)
                        a_dagger_a[p-n_mode, q-n_mode][kappa_, kappa] = a_dagger_a[q-n_mode, p-n_mode][kappa, kappa_] = p1 * p2
    
                    if MO_p == MO_q:  # <=== Necessary to build the Spins operator but not really for Hamiltonians
    
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
                                        v_term=None):
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
    
    # Building the N-electron Fermi-Hubbard matrix hamiltonian (Sparse)
    H_fermi_hubbard = scipy.sparse.csr_matrix((dim_H, dim_H))
    for p in range(n_mo):
        for q in range(n_mo):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]
            H_fermi_hubbard += E_[p, q] * h_fermion[p, q]
            for r in range(n_mo):
                for s in range(n_mo):
                    if U_fermion[p, q, r, s] != 0:  # if U is 0, it doesn't make sense to multiply matrices
                        H_fermi_hubbard += a_dagger_a[2 * p, 2 * q] @ a_dagger_a[2 * r + 1, 2 * s + 1] * U_fermion[p, q, r, s]
                        
    if v_term is not None:
        for p in range(n_mo):
            for q in range(n_mo):
                for r in range(n_mo):
                    for s in range(n_mo):
                        if v_term[p, q, r, s] != 0:  # if U is 0, it doesn't make sense to multiply matrices
                            H_fermi_hubbard += E_[p, q] @ E_[r, s] * v_term[p, q, r, s]

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
        for q in range(n_mode): 
            H_boson +=  b[p].T @ b[q] * h_boson[p,q]
     
            
    return H_fermi_hubbard + H_boson + H_fermion_boson








