import scipy
import numpy as np
from itertools import combinations_with_replacement
from numba import njit, prange
import scipy.sparse

# =============================================================================
# CORE FUNCTIONS FOR THE BUILDING OF the "A_dagger A" OPERATOR
# =============================================================================

def build_nbody_basis( n_mode, list_n_boson ):
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
    nbody_basis = []
    for n_boson in list_n_boson:
        for combination in combinations_with_replacement( range(n_mode), n_boson ):
            fock_state = [ 0 for i in range(n_mode) ]
            for index in list(combination):
                fock_state[ index ] += 1
            nbody_basis += [ fock_state ]

    return np.array(nbody_basis)  # If pybind11 is used it is better to set dtype=np.int8


# OLDER and MORE "BRUTE FORCE" FUNCTION KEPT FOR DEBUGGING
# @njit
def build_final_state_a(ref_state, p, nbodybasis):
    state_one, coeff1 = new_state_after_sq_boson_op('a',  p, ref_state) 
    kappa_ = nbodybasis.index( state_one.tolist() )
    return kappa_, coeff1


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
    if type_of_op == 'a':
        new_fock_state[index_mode] += -1
        coeff = np.sqrt( num_boson_in_mode )
    elif type_of_op == 'a^':
        new_fock_state[index_mode] += 1
        coeff = np.sqrt( num_boson_in_mode + 1)

    return new_fock_state, coeff


# numba -> njit version of build_operator_a 
def build_anihilation_operator_a(nbodybasis, silent=True):
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
    n_mode = len(nbodybasis[0])
    n_boson= np.sum(nbodybasis[0])
    # mapping_kappa = build_mapping(nbodybasis) # <== To be clearly improved

    a = np.zeros((n_mode), dtype=object)

    for p in range( n_mode ): 
        a[p] = scipy.sparse.lil_matrix((dim_H, dim_H)) 

    for p in (range(n_mode)): 
        for kappa in range(dim_H): 
            ref_state = nbodybasis[kappa]
            if  (ref_state[p] == 0):
                pass
            else : 
                kappa_, coeff1 = build_final_state_a(np.array(ref_state), p, nbodybasis.tolist()) # #  OLDER FUNCTION KEPT FOR DEBUGGING
                a[p][kappa_, kappa] =  coeff1 

    if not silent:
        print()
        print('\t ===========================================')
        print('\t ====  The matrix form of a^a is built  ====')
        print('\t ===========================================')

    return a 

 
def my_state(fockstate, nbodybasis):
    """
    Translate a fockstate (occupation number list) into a many-body
    state referenced into a given many-body basis.

    Parameters
    ----------
    fock_state  : array
        list of occupation number in each mode
    nbodybasis : array
        List of many-body states (occupation number states)

    Returns
    -------
    state :  array
        The fockstate referenced in the many-body basis
    """
    kappa = np.flatnonzero((nbodybasis == fockstate).all(1))[0]
    state = np.zeros(np.shape(nbodybasis)[0])
    state[kappa] = 1.

    return state

# =============================================================================
#  MANY-BODY HAMILTONIAN : BOSE-HUBBARD
# =============================================================================

def build_hamiltonian_bose_hubbard(h_, U_, nbodybasis, a_dagger_a ):
    """
    Create a matrix representation of the Fermi-Hubbard Hamiltonian in any
    extended many-body basis.

    Parameters
    ----------
    h_          :  array
        One-body integrals
    U_          :  array
        Two-body integrals
    nbody_basis :  array
        List of many-body states (occupation number states)
    a_dagger_a  :  array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    H_bose_hubbard :  array
        Matrix representation of the Bose-Hubbard Hamiltonian in the many-body basis
    """
    # # Dimension of the problem
    dim_H = len(nbodybasis)
    n_mode = np.shape(h_)[0]

    # Building the N-electron Fermi-Hubbard matrix hamiltonian (Sparse)
    H_bose_hubbard = scipy.sparse.csr_matrix((dim_H, dim_H))
    for p in range(n_mode):
        # H_bose_hubbard += a_dagger_a[ p, p] @  ( a_dagger_a[ p , p ] - scipy.sparse.identity(dim_H) ) * U_[p, p, p, p]
        for q in range(n_mode):
            H_bose_hubbard += a_dagger_a[p, q] * h_[p, q]

            for r in range(n_mode):
                for s in range(n_mode):
                    if U_[p, q, r, s] != 0:  # if U is 0, it doesn't make sense to multiply matrices
                        H_bose_hubbard += a_dagger_a[ p, r ] @  a_dagger_a[ q , s ]  * U_[p, q, r, s]
                        if ( q == r ):
                           H_bose_hubbard +=  - a_dagger_a[ p , s ]  * U_[p, q, r, s]


    return H_bose_hubbard


# =============================================================================
#  DIFFERENT TYPES OF REDUCED DENSITY-MATRICES
# =============================================================================

def build_1rdm(WFT, a_dagger_a):
    """
    Create a 1 RDM for a given wave function associated to a Bose-Hubbard system

    Parameters
    ----------
    WFT        :  array
        Wave function for which we want to build the 1-RDM
    a_dagger_a :  array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    one_rdm : array
        1-RDM (Bose-Hubbard system)

    """
    n_mode = np.shape(a_dagger_a)[0]
    one_rdm = np.zeros((n_mode, n_mode))
    for p in range(n_mode):
        for q in range(p, n_mode):
            one_rdm[p, q] = WFT.T @ a_dagger_a[ p, q ] @ WFT
            one_rdm[q, p] = one_rdm[p, q]
    return one_rdm


def build_2rdm(WFT, a_dagger_a):
    """
    Create a 2 RDM for a given wave function associated to a Bose-Hubbard system

    Parameters
    ----------
    WFT        : array
        Wave function for which we want to build the 1-RDM
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator

    Returns
    -------
    two_rdm : array
        2-RDM (Bose-Hubbard system)

    """
    n_mode = np.shape(a_dagger_a)[0]
    two_rdm = np.zeros((n_mode, n_mode, n_mode, n_mode))
    for p in range(n_mode):
        for q in range(n_mode):
            for r in range(n_mode):
                for s in range(n_mode):
                    two_rdm[p, q, r, s] += WFT.T @ a_dagger_a[ p, r ] @ a_dagger_a[ q, s ]   @ WFT
                    if ( q == r ):
                        two_rdm[p, q, r, s] += - WFT.T @  a_dagger_a[ p , s ]  @ WFT
    return two_rdm



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
        1-boson integrals given in basis B1
    g_b1 : array
        2-boson integrals given in basis B1
    C    : array
        Transfer matrix

    Returns
    -------
    h_b2 : array
        1-boson integrals given in basis B2
    g_b2 : array
        2-boson integrals given in basis B2
    """
    h_b2 = np.einsum('pi,qj,pq->ij', C, C, h_b1, optimize=True)
    g_b2 = np.einsum('ap, bq, cr, ds, abcd -> pqrs', C, C, C, C, g_b1, optimize=True)

    return h_b2, g_b2


# =============================================================================
#  FUNCTION TO HELP THE VISUALIZATION OF MANY-BOSON WAVE FUNCTIONS
# =============================================================================

def visualize_wft(WFT, nbodybasis, cutoff=0.005 ):
    """
    Print the decomposition of a given input wave function in a many-body basis.

    Parameters
    ----------
    WFT              : array
        Reference wave function
    nbody_basis      : array
        List of many-body states (occupation number states)
    cutoff           : array
        Cut off for the amplitudes retained (default is 0.005)

    Returns
    -------
        Terminal printing of the wave function

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
