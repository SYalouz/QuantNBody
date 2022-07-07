import scipy
import numpy as np
from itertools import combinations_with_replacement
from numba import njit, prange 
import scipy.sparse
from tqdm import tqdm 
 
# =============================================================================
# CORE FUNCTIONS FOR THE BUILDING OF the "A_dagger A" OPERATOR
# =============================================================================

def build_nbody_basis( n_mode, n_boson ):
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
    max_number = n_boson * 10**(( n_mode - 1 )*2) + 1
    dim_H = np.shape(nbodybasis)[0] 
    mapping_kappa = np.zeros( max_number, dtype=np.int32 )
    for kappa in range(dim_H):
        ref_state = nbodybasis[ kappa ]
        number = 0 
        for index_mode in range(n_mode):  
            number += ref_state[index_mode] * 10**(( n_mode - index_mode - 1 )*2)
        mapping_kappa[ int(np.round(number)) ] = kappa
        
    return mapping_kappa


# @njit
# def build_final_state_ad_a(ref_state, p, q, mapping_kappa):
#     state_one, coeff1 = new_state_after_sq_boson_op('a',  q, ref_state)
#     state_two, coeff2 = new_state_after_sq_boson_op('a^', p, state_one)
#     kappa_ = mapping_kappa[ make_number_out_of_vector(state_two) ]  
 
#     return kappa_, coeff1, coeff2
 

# OLDER and MORE "BRUTE FORCE" FUNCTION KEPT FOR DEBUGGING 
# @njit
def build_final_state_ad_a(ref_state, p, q, nbodybasis):
    state_one, coeff1 = new_state_after_sq_boson_op('a',  q, ref_state)
    state_two, coeff2 = new_state_after_sq_boson_op('a^', p, state_one) 
    kappa_ = nbodybasis.index( state_two.tolist() ) 
    return kappa_, coeff1, coeff2


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
    # mapping_kappa = build_mapping(nbodybasis) # <== To be clearly improved

    a_dagger_a = np.zeros((n_mode, n_mode), dtype=object)
   
    for p in range( n_mode ): 
        for q in range(p, n_mode):
            a_dagger_a[p, q] = scipy.sparse.lil_matrix((dim_H, dim_H))
            a_dagger_a[q, p] = scipy.sparse.lil_matrix((dim_H, dim_H))
     
    for q in tqdm(range(n_mode)): 
        for p in range(q, n_mode): 
            for kappa in range(dim_H):
                
                ref_state = nbodybasis[kappa]
                if p != q and (ref_state[q] == 0 or ref_state[p] == n_boson): 
                    pass 
                else :
                    # kappa_, coeff1, coeff2 = build_final_state_ad_a(np.array(ref_state), p, q, mapping_kappa)
                    kappa_, coeff1, coeff2 = build_final_state_ad_a(np.array(ref_state), p, q, nbodybasis.tolist()) # #  OLDER FUNCTION KEPT FOR DEBUGGING 
                    a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] =  coeff1 * coeff2
                    
    if not silent:
        print()
        print('\t ===========================================')
        print('\t ====  The matrix form of a^a is built  ====')
        print('\t ===========================================')

    return a_dagger_a
 

@njit
def make_number_out_of_vector( ref_state ):
    """
    Function to translate a single many-body configuration into an unique integer

    Parameters
    ----------
    ref_state : Reference slater determinant to turn out into an integer

    Returns
    -------
    number : unique integer referring to the Fock state
    """ 
    # print( ref_state )
    n_mode  = len(ref_state) 
    n_boson = np.sum(ref_state)  
    num_digits = 0 
    while n_boson != 0:
        n_boson //= 10
        num_digits += 1
    # print('numdigit',num_digits )
    number = 0
    for index_mode in range(n_mode):
        number += ref_state[index_mode] * 10**(( n_mode - index_mode - 1 )*2) #<= New type of counting for boson
    # print('number',number )    
    
    return number 

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
    v_term      :  4D matrix that is already transformed into correct representation.

    Returns
    -------
    H_bose_hubbard :  Matrix representation of the Bose-Hubbard Hamiltonian
    """
    # # Dimension of the problem 
    dim_H = len(nbodybasis)
    n_mode = np.shape(h_)[0]

    # Building the N-electron Fermi-Hubbard matrix hamiltonian (Sparse)
    H_bose_hubbard = scipy.sparse.csr_matrix((dim_H, dim_H))
    for p in range(n_mode):
        H_bose_hubbard += a_dagger_a[ p, p] @  ( a_dagger_a[ p , p ] - scipy.sparse.identity(dim_H) ) * U_[p, p, p, p]
        for q in range(n_mode): 
            H_bose_hubbard += a_dagger_a[p, q] * h_[p, q]
            
            # for r in range(n_mode):
            #     for s in range(n_mode):
            #         if U_[p, q, r, s] != 0:  # if U is 0, it doesn't make sense to multiply matrices
            #             H_bose_hubbard += a_dagger_a[ p, q] @  a_dagger_a[ r , s ]   * U_[p, q, r, s]
            #             H_bose_hubbard += (a_dagger_a[ p, q] @ 
            #                               (a_dagger_a[ r , s ] - scipy.sparse.identity(dim_H)) ) * U_[p, q, r, s]
                        
    # if v_term is not None:
    #     for p in range(n_mo):
    #         for q in range(n_mo):
    #             for r in range(n_mo):
    #                 for s in range(n_mo):
    #                     if v_term[p, q, r, s] != 0:  # if U is 0, it doesn't make sense to multiply matrices
    #                         H_fermi_hubbard += E_[p, q] @ E_[r, s] * v_term[p, q, r, s]
                            
    return H_bose_hubbard
 
    
# =============================================================================
#  DIFFERENT TYPES OF REDUCED DENSITY-MATRICES
# =============================================================================

def build_1rdm(WFT, a_dagger_a):
    """
    Create a 1 RDM out of a given wave function

    Parameters
    ----------
    WFT        :  Wave function for which we want to build the 1-RDM
    a_dagger_a :  Matrix representation of the a_dagger_a operator

    Returns
    -------
    One_Rone_rdmDM : 1-RDM

    """
    n_mode = np.shape(a_dagger_a)[0]
    one_rdm = np.zeros((n_mode, n_mode))
    for p in range(n_mode):
        for q in range(p, n_mode):
            one_rdm[p, q] = WFT.T @ a_dagger_a[ p, q ] @ WFT
            one_rdm[q, p] = one_rdm[p, q]
    return one_rdm

    
# =============================================================================
#  FUNCTION TO HELP THE VISUALIZATION OF MANY-BOSON WAVE FUNCTIONS
# =============================================================================

def visualize_wft(WFT, nbodybasis, cutoff=0.005 ):
    """
    Print the decomposition of a given input wave function in a many-body basis.

   Parameters
    ----------
    WFT              : Reference wave function
    nbody_basis      : List of many-body states (occupation number states)
    cutoff           : Cut off for the amplitudes retained (default is 0.005) 

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

