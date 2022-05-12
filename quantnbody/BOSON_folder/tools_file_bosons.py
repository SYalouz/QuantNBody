import scipy
import numpy as np
from itertools import combinations_with_replacement
from numba import njit, prange 
import scipy.sparse

E_ = False
e_ = False
 
# =============================================================================
# CORE FUNCTIONS FOR THE BUILDING OF the "A_dagger A" OPERATOR
# =============================================================================

def build_nbody_basis(n_mode, n_boson, S_z_cleaning=False):
    """
    Create a many-body basis formed by a list of fock-state with a conserved total
    number of bosons 

    Parameters
    ----------
    n_mode    :  Number of modes in total 
    N_boson   :  Number of bosons in total 

    Returns
    -------
    nbody_basis :  List of many-body states (occupation number states)  
    """
    # Building the N-electron many-body basis
    nbody_basis = []
    for combination in combinations_with_replacement( range(n_mode), n_boson ):
        fock_state = [ 0 for i in range(n_mode) ]
        for index in list(combination):
            fock_state[ index ] += 1
        nbody_basis += [ fock_state ]
        
    return np.array(nbody_basis)  # If pybind11 is used it is better to set dtype=np.int8

# =====
# OK  !!!!
# =====

@njit
def build_mapping( nbodybasis ):
    """
    Function to create a unique mapping between a kappa vector and an occupation
    number state.

    Parameters
    ----------
    nbody_basis :  Many-body basis

    Returns
    -------
    mapping_kappa : List of unique values associated to each kappa
    """
    n_mode  = np.shape(nbodybasis)[1]
    n_boson = np.sum(nbodybasis[0])
    max_number = n_boson * 10**(n_mode-1)
    dim_H = np.shape(nbodybasis)[0] 
    mapping_kappa = np.zeros( max_number, dtype=np.int32 )
    for kappa in range(dim_H):
        ref_state = nbodybasis[ kappa ]
        number = 0 
        for index_mode in range(n_mode):  
            number += ref_state[index_mode] * 10**(n_mode - index_mode - 1)
        mapping_kappa[number] = kappa
        
    return mapping_kappa

# =====
# OK  !!!!
# =====


@njit
def build_final_state_ad_a(ref_state, p, q, mapping_kappa):
    state_one, coeff1 = new_state_after_sq_boson_op('a',  q, ref_state)
    state_two, coeff2 = new_state_after_sq_boson_op('a^', p, state_one)
    kappa_ = mapping_kappa[make_number_out_of_vector(state_two)]
 
    return kappa_, coeff1,  coeff2

# =====
# OK  !!!!
# =====

@njit
def new_state_after_sq_boson_op(type_of_op, index_mode, ref_fock_state):
    """
    Parameters
    ----------
    type_of_op    :  type of operator to apply (creation of annihilation)
    index_mode    :  index of the second quantized mode to occupy/empty
    ref_fock_state :  initial state to be transformed

    Returns
    -------
    new_fock_state :  Resulting occupation number form of the transformed state
    coeff_phase   :  Phase attached to the resulting state

    """
    new_fock_state    = ref_fock_state.copy()
    num_boson_in_mode = ref_fock_state[index_mode]
    if type_of_op == 'a':
        new_fock_state[index_mode] += -1
        coeff = np.sqrt( num_boson_in_mode )
    elif type_of_op == 'a^':
        new_fock_state[index_mode] += 1
        coeff = np.sqrt( num_boson_in_mode + 1)

    return new_fock_state, coeff

# =====
# OK  !!!!
# =====

# numba -> njit version of build_operator_a_dagger_a
def build_operator_a_dagger_a(nbodybasis, silent=True):
    """
    Create a matrix representation of the a_dagger_a operator
    in the many-body basis

    Parameters
    ----------
    nbody_basis :  List of many-body states (occupation number states) (occupation number states)
    silent      :  If it is True, function doesn't print anything when it generates a_dagger_a
    Returns
    -------
    a_dagger_a :  Matrix representation of the a_dagger_a operator

    """
    # Dimensions of problem
    dim_H  = len(nbodybasis)
    n_mode = len(nbodybasis[0]) 
    n_boson= np.sum(nbodybasis[0])
    mapping_kappa = build_mapping(nbodybasis)

    a_dagger_a = np.zeros((n_mode, n_mode), dtype=object)
    for p in range( n_mode ):
        for q in range(p, n_mode):
            a_dagger_a[p, q] = scipy.sparse.lil_matrix((dim_H, dim_H))
            a_dagger_a[q, p] = scipy.sparse.lil_matrix((dim_H, dim_H))

    for q in (range(n_mode)):
        for p in range(q, n_mode):
            for kappa in range(dim_H):
                ref_state = nbodybasis[kappa]
                if p != q and (ref_state[q] == 0 or ref_state[p] == n_boson): 
                    pass
                else :
                    kappa_, coeff1, coeff2 = build_final_state_ad_a(np.array(ref_state), p, q, mapping_kappa)
                    a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] =  coeff1 * coeff2

    if not silent:
        print()
        print('\t ===========================================')
        print('\t ====  The matrix form of a^a is built  ====')
        print('\t ===========================================')

    return a_dagger_a

# =====
# OK  !!!!
# =====

@njit
def make_number_out_of_vector(ref_state):
    """
    Function to translate a slater determinant into an unique integer

    Parameters
    ----------
    ref_state : Reference slater determinant to turn out into an integer

    Returns
    -------
    number : unique integer referring to the Fock state
    """
    n_mode  = len(ref_state) 
    number = 0
    for index_mode in range(n_mode):
        number += ref_state[index_mode] * 10**(n_mode - index_mode - 1)
    return number

# =====
# OK  !!!!
# =====


def my_state(fockstate, nbodybasis):
    """
    Translate a Slater determinant (occupation number list) into a many-body
    state referenced into a given Many-body basis.

    Parameters
    ----------
    fock_state  : occupation number list
    nbodybasis : List of many-body states (occupation number states)

    Returns
    -------
    state :  The slater determinant referenced in the many-body basis
    """
    kappa = np.flatnonzero((nbodybasis == fockstate).all(1))[0]  
    state = np.zeros(np.shape(nbodybasis)[0])
    state[kappa] = 1.

    return state

# =====
# OK  !!!!
# =====


# =============================================================================
#  MANY-BODY HAMILTONIAN : BOSE-HUBBARD 
# =============================================================================


def build_hamiltonian_bose_hubbard(h_, U_, nbodybasis, a_dagger_a ):
    """
    Create a matrix representation of the Fermi-Hubbard Hamiltonian in any
    extended many-body basis.

    Parameters
    ----------
    h_          :  One-body integrals
    U_          :  Two-body integrals
    nbody_basis :  List of many-body states (occupation number states)
    a_dagger_a  :  Matrix representation of the a_dagger_a operator
    S_2         :  Matrix representation of the S_2 operator (default is None)
    S_2_target  :  Value of the S_2 mean value we want to target (default is None)
    penalty     :  Value of the penalty term for state not respecting the spin symmetry (default is 100).
    v_term      :  4D matrix that is already transformed into correct representation.

    Returns
    -------
    H_bose_hubbard :  Matrix representation of the Fermi-Hubbard Hamiltonian

    """
    # # Dimension of the problem 
    dim_H = len(nbodybasis)
    n_mo = np.shape(h_)[0]

    # Building the N-electron Fermi-Hubbard matrix hamiltonian (Sparse)
    H_bose_hubbard = scipy.sparse.csr_matrix((dim_H, dim_H))
    for p in range(n_mo):
        for q in range(n_mo): 
            H_bose_hubbard += a_dagger_a[p, q] * h_[p, q]
            for r in range(n_mo):
                for s in range(n_mo):
                    if U_[p, q, r, s] != 0:  # if U is 0, it doesn't make sense to multiply matrices
                        H_bose_hubbard += (a_dagger_a[ p, q] @ 
                                           (a_dagger_a[ r , s ] - scipy.sparse.identity(dim_H)) ) * U_[p, q, r, s]
                        
    # if v_term is not None:
    #     for p in range(n_mo):
    #         for q in range(n_mo):
    #             for r in range(n_mo):
    #                 for s in range(n_mo):
    #                     if v_term[p, q, r, s] != 0:  # if U is 0, it doesn't make sense to multiply matrices
    #                         H_fermi_hubbard += E_[p, q] @ E_[r, s] * v_term[p, q, r, s]
                            
    return H_bose_hubbard




# =====
# OK  !!!!
# =====


# =============================================================================
#  DIFFERENT TYPES OF REDUCED DENSITY-MATRICES
# =============================================================================


def build_full_mo_1rdm_and_2rdm( WFT, a_dagger_a, active_indices, n_mo_total ):
    """
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build the MO 1/2-ELECTRON DENSITY MATRICES from a
    reference wavefunction expressed in the computational basis
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    """ 
    n_mo = len( active_indices )
    one_rdm_a = np.zeros((n_mo_total, n_mo_total))
    two_rdm_a = np.zeros((n_mo_total, n_mo_total, n_mo_total, n_mo_total))
    first_act_index = active_indices[0]
    
    global E_
    global e_
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
                        
    # Creating RDMs elements only within the frozen space
    if first_act_index > 0:
        for i in range(first_act_index):
            for j in range(first_act_index):
                one_rdm_a[i, j] = 2. * delta(i, j)
                for k in range(first_act_index):
                    for l in range(first_act_index):
                        # State A
                        two_rdm_a[i, j, k, l] = 4. * delta(i, j) * delta(k, l) - 2. * delta(i, l) * delta(j, k)

    # Creating RDMs elements in the the active/frozen spaces
    for p in active_indices:
        for q in active_indices:
            # Shifting the indices
            p_ = p - first_act_index
            q_ = q - first_act_index 
            # 1-RDM elements only within the active space
            # State A
            one_rdm_a[p, q] = WFT.T @ E_[p_, q_] @ WFT 

            # 2-RDM elements only within the active space
            for r in active_indices:
                for s in active_indices:
                    # Shifting the indices
                    r_ = r - first_act_index
                    s_ = s - first_act_index 
                    # State A
                    two_rdm_a[p, q, r, s] = WFT.T @ e_[p_, q_, r_, s_] @ WFT 

            if first_act_index > 0:
                # 2-RDM elements between the active and frozen spaces
                for i in range(first_act_index):
                    for j in range(first_act_index):
                        # State A
                        two_rdm_a[i, j, p, q] = two_rdm_a[p, q, i, j] = 2. * delta(i, j) * one_rdm_a[p, q]
                        two_rdm_a[p, i, j, q] = two_rdm_a[j, q, p, i] = - delta(i, j) * one_rdm_a[p, q]

    return one_rdm_a, two_rdm_a



def build_1rdm(WFT, a_dagger_a):
    """
    Create a spin-alpha 1 RDM out of a given wave function

    Parameters
    ----------
    WFT        :  Wave function for which we want to build the 1-RDM
    a_dagger_a :  Matrix representation of the a_dagger_a operator

    Returns
    -------
    One_RDM_alpha : spin-alpha 1-RDM

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    one_rdm_alpha = np.zeros((n_mo, n_mo))
    for p in range(n_mo):
        for q in range(p, n_mo):
            one_rdm_alpha[p, q] = WFT.T @ a_dagger_a[2 * p, 2 * q] @ WFT
            one_rdm_alpha[q, p] = one_rdm_alpha[p, q]
    return one_rdm_alpha



def build_2rdm_fh_on_site_repulsion(WFT, a_dagger_a, mask=None):
    """
    Create a spin-free 2 RDM out of a given Fermi Hubbard wave function for the on-site repulsion operator.
    (u[i,j,k,l] corresponds to a^+_i↑ a_j↑ a^+_k↓ a_l↓)

    Parameters
    ----------
    WFT        :  Wave function for which we want to build the 1-RDM
    a_dagger_a :  Matrix representation of the a_dagger_a operator
    mask       :  4D array is expected. Function is going to calculate only elements of 2rdm where mask is not 0.
                  For default None the whole 2RDM is calculated.
                  If we expect 2RDM to be very sparse (has only a few non-zero elements) then it is better to provide
                  array that ensures that we won't calculate elements that are not going to be used in calculation of
                  2-electron interactions.
    Returns
    -------
    two_rdm for the on-site-repulsion operator
    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    two_rdm_fh = np.zeros((n_mo, n_mo, n_mo, n_mo))
    for p in range(n_mo):
        for q in range(n_mo):
            for r in range(n_mo):
                for s in range(n_mo):
                    if mask is None or mask[p, q, r, s] != 0:
                        two_rdm_fh[p, q, r, s] += WFT.T @ a_dagger_a[2 * p, 2 * q] @ a_dagger_a[2 * r + 1, 2 * s + 1] @ WFT
    return two_rdm_fh






# =============================================================================
#   INTEGRALS BUILDER FOR ACTIVE SPACE CALCULATION
# =============================================================================

def fh_get_active_space_integrals(h_, U_, frozen_indices=None, active_indices=None):
    """
    Restricts a Fermi-Hubbard at a spatial orbital level to an active space
    This active space may be defined by a list of active indices and
    doubly occupied indices. Note that one_body_integrals and
    two_body_integrals must be defined in an orthonormal basis set (MO like).
    Args:
         - occupied_indices: A list of spatial orbital indices
           indicating which orbitals should be considered doubly occupied.
         - active_indices: A list of spatial orbital indices indicating
           which orbitals should be considered active.
         - 1 and 2 body integrals.
    Returns:
        tuple: Tuple with the following entries:
        **core_constant**: Adjustment to constant shift in Hamiltonian
        from integrating out core orbitals
        **one_body_integrals_new**: one-electron integrals over active space.
        **two_body_integrals_new**: two-electron integrals over active space.
    """
    # Determine core Energy from frozen MOs
    core_energy = 0
    for i in frozen_indices:
        core_energy += 2 * h_[i, i]
        for j in frozen_indices:
            core_energy += U_[i, i, j, j]

    # Modified one-electron integrals
    h_act = h_.copy()
    for t in active_indices:
        for u in active_indices:
            for i in frozen_indices:
                h_act[t, u] += U_[i, i, t, u]

    return (core_energy,
            h_act[np.ix_(active_indices, active_indices)],
            U_[np.ix_(active_indices, active_indices, active_indices, active_indices)])


def qc_get_active_space_integrals(one_body_integrals,
                                  two_body_integrals,
                                  occupied_indices=None,
                                  active_indices=None):
    """
        >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        Restricts a Quantum chemistry Hamiltonian at a spatial orbital level
        to an active space. This active space may be defined by a list of
        active indices and doubly occupied indices. Note that one_body_integrals and
        two_body_integrals must be defined in an orthonormal basis set (MO like).
        Args:
              - occupied_indices: A list of spatial orbital indices
                indicating which orbitals should be considered doubly occupied.
              - active_indices: A list of spatial orbital indices indicating
                which orbitals should be considered active.
              - 1 and 2 body integrals.
        Returns:
            tuple: Tuple with the following entries:
            **core_constant**: Adjustment to constant shift in Hamiltonian
            from integrating out core orbitals
            one_body_integrals_new : one-electron integrals over active space.
            two_body_integrals_new : two-electron integrals over active space.
        >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        """
    # Fix data type for a few edge cases
    occupied_indices = [] if occupied_indices is None else occupied_indices
    if len(active_indices) < 1:
        raise ValueError('Some active indices required for reduction.')

    # Determine core constant
    core_constant = 0.0
    for i in occupied_indices:
        core_constant += 2 * one_body_integrals[i, i]
        for j in occupied_indices:
            core_constant += (2 * two_body_integrals[i, i, j, j]
                              - two_body_integrals[i, j, j, i])
            # core_constant += (2 * two_body_integrals[i, j, i, j]
            #                   - two_body_integrals[i, j, j, i])

    # Modified one electron integrals
    one_body_integrals_new = np.copy(one_body_integrals)
    for u in active_indices:
        for v in active_indices:
            for i in occupied_indices:
                one_body_integrals_new[u, v] += ( 2 * two_body_integrals[i, i, u, v]
                                                  - two_body_integrals[i, u, v, i] )

    # Restrict integral ranges and change M appropriately
    return (core_constant,
            one_body_integrals_new[np.ix_(active_indices, active_indices)],
            two_body_integrals[np.ix_(active_indices, active_indices, active_indices, active_indices)])


# =============================================================================
#  FUNCTION TO HELP THE VISUALIZATION OF MANY-BODY WAVE FUNCTIONS
# =============================================================================

def visualize_wft(WFT, nbodybasis, cutoff=0.005, atomic_orbitals=False):
    """
    Print the decomposition of a given input wave function in a many-body basis.

   Parameters
    ----------
    WFT              : Reference wave function
    nbody_basis      : List of many-body states (occupation number states)
    cutoff           : Cut off for the amplitudes retained (default is 0.005)
    atomic_orbitals  : Boolean; If True then instead of 0/1 for spin orbitals we get 0/alpha/beta/2 for atomic orbitals

    Returns
    -------
    Same string that was printed to the terminal (the wave function)

    """
    list_index = np.where(abs(WFT) > cutoff)[0]

    states = []
    coefficients = []
    for index in list_index:
        coefficients += [WFT[index]]
        states += [nbodybasis[index]]

    list_sorted_index = np.flip(np.argsort(np.abs(coefficients)))

    return_string = f'\n\t{"-" * 11}\n\t Coeff.      N-body state\n\t{"-" * 7}     {"-" * 13}\n'
    for index in list_sorted_index[0:8]:
        state = states[index]

        ket = '|' +   ",".join([str(elem) for elem in state]) + '⟩'
        return_string += f'\t{coefficients[index]:+1.5f}\t{ket}\n'
        
    print(return_string)
    
    return return_string


@njit
def delta(index_1, index_2):
    """
    Function delta kronecker
    """
    d = 0.0
    if index_1 == index_2:
        d = 1.0
    return d

