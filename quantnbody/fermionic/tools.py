import scipy
from scipy.optimize import minimize
import numpy as np
from itertools import combinations
from numba import njit, prange
import matplotlib.pyplot as plt
import psi4

E_ = False
e_ = False

# =============================================================================
# CORE FUNCTIONS FOR THE BUILDING OF the "A_dagger A" OPERATOR
# =============================================================================

# def build_nbody_basis(n_mo, N_electron, S_z_cleaning=False):
#     """
#     Create a many-body basis as a list of slater-determinants. Here, these states are
#     occupation numbersvectors taking the form of bitstrings (*e.g.* \|1100⟩)
#     describing how the electrons occupy the spin-orbitals.

#     Parameters
#     ----------
#     n_mo : int
#         Number of molecular orbitals
#     N_electron :  int
#         Number of electrons
#     S_z_cleaning : bool, default=False
#         Option if we want to get rid of the s_z != 0 states (default is False)

#     Returns
#     -------
#     nbody_basis : array
#         List of many-body states (occupation number vectors).

#     Examples
#     ________
#     >>> build_nbody_basis(2, 2, False)  # 2 electrons in 2 molecular orbitals
#     array([[1, 1, 0, 0],
#            [1, 0, 1, 0],
#            [1, 0, 0, 1],
#            [0, 1, 1, 0],
#            [0, 1, 0, 1],
#            [0, 0, 1, 1]])

#     """
#     # Building the N-electron many-body basis
#     nbody_basis = []
#     for combination in combinations(range(2 * n_mo), N_electron):
#         fock_state = [0] * (2 * n_mo)
#         for index in list(combination):
#             fock_state[index] += 1
#         nbody_basis += [fock_state]

#     # In case we want to get rid of states with s_z != 0
#     if S_z_cleaning:
#         nbody_basis_cleaned = nbody_basis.copy()
#         for i in range(np.shape(nbody_basis)[0]):
#             s_z = check_sz(nbody_basis[i])
#             if s_z != 0:
#                 nbody_basis_cleaned.remove(nbody_basis[i])
#         nbody_basis = nbody_basis_cleaned

#     return np.array(nbody_basis)  # If pybind11 is used it is better to set dtype=np.int8

def build_nbody_basis(n_mo, N_electron, S_z_cleaning=False):
    """
    Create a many-body basis as a list of slater-determinants. Here, these states are
    occupation numbersvectors taking the form of bitstrings (*e.g.* \|1100⟩)
    describing how the electrons occupy the spin-orbitals.

    Parameters
    ----------
    n_mo : int
        Number of molecular orbitals
    N_electron :  int OR list
        Number of electrons or list of number of electrons
    S_z_cleaning : bool, default=False
        Option if we want to get rid of the s_z != 0 states (default is False)

    Returns
    -------
    nbody_basis : array
        List of many-body states (occupation number vectors).

    Examples
    ________
    >>> build_nbody_basis(2, 2, False)  # 2 electrons in 2 molecular orbitals
    array([[1, 1, 0, 0],
           [1, 0, 1, 0],
           [1, 0, 0, 1],
           [0, 1, 1, 0],
           [0, 1, 0, 1],
           [0, 0, 1, 1]])

    """
    # Building the N-electron many-body basis
    nbody_basis = [] 
    if type( N_electron ) is int : 
        for combination in combinations(range(2 * n_mo), N_electron):
            fock_state = [0] * (2 * n_mo)
            for index in list(combination):
                fock_state[index] += 1
            nbody_basis += [fock_state]  
    else:
        for N_elec in N_electron: 
            for combination in combinations(range(2 * n_mo), N_elec):
                fock_state = [0] * (2 * n_mo)
                for index in list(combination):
                    fock_state[index] += 1
                nbody_basis += [fock_state]

    # In case we want to get rid of states with s_z != 0
    if S_z_cleaning:
        nbody_basis_cleaned = nbody_basis.copy()
        for i in range(np.shape(nbody_basis)[0]):
            s_z = check_sz(nbody_basis[i])
            if s_z != 0:
                nbody_basis_cleaned.remove(nbody_basis[i])
        nbody_basis = nbody_basis_cleaned

    return np.array(nbody_basis)  # If pybind11 is used it is better to set dtype=np.int8

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


# numba -> njit version of build_operator_a_dagger_a
def build_operator_a_dagger_a(nbody_basis, silent=True):
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
    n_mo = len(nbody_basis[0]) // 2
    mapping_kappa = build_mapping(nbody_basis)

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
                p, q = 2 * MO_p, 2 * MO_q
                if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
                    pass
                elif ref_state[q] == 1:
                    kappa_, p1, p2 = build_final_state_ad_a(np.array(ref_state), p, q, mapping_kappa)
                    a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] = p1 * p2

                # Single excitation : spin beta -- beta
                p, q = 2 * MO_p + 1, 2 * MO_q + 1
                if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
                    pass
                elif ref_state[q] == 1:
                    kappa_, p1, p2 = build_final_state_ad_a(np.array(ref_state), p, q, mapping_kappa)
                    a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] = p1 * p2

                # Single excitation : spin beta -- alpha
                p, q = 2 * MO_p + 1, 2 * MO_q
                if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
                    pass
                elif ref_state[q] == 1:
                    kappa_, p1, p2 = build_final_state_ad_a(np.array(ref_state), p, q, mapping_kappa)
                    a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] = p1 * p2

                # Single excitation : spin alpha -- beta
                p, q = 2 * MO_p, 2 * MO_q + 1 
                if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
                    pass
                elif ref_state[q] == 1:
                    kappa_, p1, p2 = build_final_state_ad_a(np.array(ref_state), p, q, mapping_kappa)
                    a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] = p1 * p2
                        
    if not silent:
        print()
        print('\t ===========================================')
        print('\t ====  The matrix form of a^a is built  ====')
        print('\t ===========================================')

    return a_dagger_a



# numba -> njit version of build_operator_a_dagger_a
def TEST_build_operator_a_dagger_a(nbody_basis, silent=True):
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
    n_mo = len(nbody_basis[0]) // 2
    mapping_kappa = build_mapping(nbody_basis)

    a_dagger_a = np.zeros((2 * n_mo, 2 * n_mo), dtype=object)
    Mat = scipy.sparse.lil_matrix((dim_H, dim_H))
    for p in range(2 * n_mo):
        for q in range(p, 2 * n_mo):
            a_dagger_a[p, q] = Mat
            a_dagger_a[q, p] = Mat #scipy.sparse.lil_matrix((dim_H, dim_H)) 

    # for MO_q in (prange(n_mo)):
    #     for MO_p in range(MO_q, n_mo):
    #         for kappa in range(dim_H):
    #             ref_state = nbody_basis[kappa]
                
    #             # Single excitation : spin alpha -- alpha
    #             p, q = 2 * MO_p, 2 * MO_q
    #             if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
    #                 pass
    #             elif ref_state[q] == 1:
    #                 kappa_, p1, p2 = build_final_state_ad_a(np.array(ref_state), p, q, mapping_kappa)
    #                 a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] = p1 * p2

    #             # Single excitation : spin beta -- beta
    #             p, q = 2 * MO_p + 1, 2 * MO_q + 1
    #             if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
    #                 pass
    #             elif ref_state[q] == 1:
    #                 kappa_, p1, p2 = build_final_state_ad_a(np.array(ref_state), p, q, mapping_kappa)
    #                 a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] = p1 * p2

    #             if MO_p == MO_q:  # <=== Necessary to build the Spins operator but not really for Hamiltonians

    #                 # Single excitation : spin beta -- alpha
    #                 p, q = 2 * MO_p + 1, 2 * MO_p
    #                 if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
    #                     pass
    #                 elif ref_state[q] == 1:
    #                     kappa_, p1, p2 = build_final_state_ad_a(np.array(ref_state), p, q, mapping_kappa)
    #                     a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] = p1 * p2

    #                     # Single excitation : spin alpha -- beta
    #                 p, q = 2 * MO_p, 2 * MO_p + 1

    #                 if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
    #                     pass
    #                 elif ref_state[q] == 1:
    #                     kappa_, p1, p2 = build_final_state_ad_a(np.array(ref_state), p, q, mapping_kappa)
    #                     a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] = p1 * p2
    # if not silent:
    #     print()
    #     print('\t ===========================================')
    #     print('\t ====  The matrix form of a^a is built  ====')
    #     print('\t ===========================================')

    return a_dagger_a

@njit
def build_mapping(nbody_basis):
    """
    Create a unique mapping between a kappa vector and an occupation
    number state. This is important to speedup the building of the a_dagger_a operator.

    Parameters
    ----------
    nbody_basis :  Iterable[Iterable[int]]
        List of many-body state written in terms of occupation numbers

    Returns
    -------
    mapping_kappa : Iterable[int]
        List of unique values associated to each kappa

    Examples
    ________
    >>> nbody_basis = build_nbody_basis(2, 2)
    >>> build_mapping(nbody_basis)
    array([0, 0, 0, 5, 0, 4, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0])

    """
    num_digits = np.shape(nbody_basis)[1]
    dim_H = np.shape(nbody_basis)[0]
    mapping_kappa = np.zeros(2 ** num_digits, dtype=np.int32)
    for kappa in range(dim_H):
        ref_state = nbody_basis[kappa]
        number = 0
        for digit in range(num_digits):
            number += ref_state[digit] * 2 ** (num_digits - digit - 1)
        mapping_kappa[number] = kappa

    return mapping_kappa


@njit
def make_integer_out_of_bit_vector(ref_state):
    """
    Translate a slater determinant written as a list of occupation number
    into an unique integer via a reversed bitstring transformation

    Parameters
    ----------
    ref_state : Iterable[int]
        Reference slater determinant to turn out into an integer

    Returns
    -------
    int
        Number of unique integer referring to the slater determinant.

    Examples
    ________
    >>> make_integer_out_of_bit_vector(np.array([1, 1, 0, 0]))
    12

    """
    number = 0
    for digit in range(len(ref_state)):
        number += ref_state[digit] * 2 ** (len(ref_state) - digit - 1)

    return number


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


@njit
def build_final_state_ad_a(ref_state, p, q, mapping_kappa):
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
    state_one, p1 = new_state_after_sq_fermi_op('a', q, ref_state)
    state_two, p2 = new_state_after_sq_fermi_op('a^', p, state_one)
    kappa_ = mapping_kappa[make_integer_out_of_bit_vector(state_two)]

    return kappa_, p1, p2


# =============================================================================
#  MANY-BODY HAMILTONIANS (FERMI HUBBARD AND QUANTUM CHEMISTRY)
# =============================================================================

def build_hamiltonian_quantum_chemistry(h_,
                                        g_,
                                        nbody_basis,
                                        a_dagger_a,
                                        S_2=None,
                                        S_2_target=None,
                                        penalty=100,
                                        cut_off_integral=1e-8):
    """
    Create a matrix representation of the electronic structure Hamiltonian in the
    many-body basis

    Parameters
    ----------
    h_ : array
        One-body integrals
    g_ : array
        Two-body integrals
    nbody_basis :  array
        List of many-body states (occupation number states)
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator
    S_2 : array, default=None
        Matrix representation of the S_2 operator (default is None)
    S_2_target : float, default=None
        Value of the S_2 mean value we want to target (default is None)
    penalty : float, default=100
        Value of the penalty term to penalize the states that do not respect the spin symmetry (default is 100).

    Returns
    -------
    H_chemistry : array
         Matrix representation of the electronic structure Hamiltonian in the many-body basis

    """
    # Dimension of the problem
    dim_H = len(nbody_basis)
    n_mo = np.shape(h_)[0]

    # Building the spin-preserving one-body excitation operator
    global E_
    global e_
    E_ = np.empty((n_mo, n_mo), dtype=object)
    e_ = np.empty((n_mo, n_mo, n_mo, n_mo), dtype=object)
    H_chemistry = scipy.sparse.csr_matrix((dim_H, dim_H))

    for p in range(n_mo):
        for q in range(n_mo):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]
            
    # for p in range(n_mo):
    #     for q in range(n_mo):
    #         H_chemistry += E_[p, q] * h_[p, q]
    #         for r in range(n_mo):
    #             for s in range(n_mo):
    #                 e_[p, q, r, s] = E_[p, q] @ E_[r, s]
    #                 if q == r:
    #                     e_[p, q, r, s] += - E_[p, s] 
    #                 H_chemistry += e_[p, q, r, s] * g_[p, q, r, s] / 2.
    
    indices_one_electron_integrals = np.transpose((abs(h_)>cut_off_integral).nonzero())
    for indices in indices_one_electron_integrals:
        p, q = indices 
        H_chemistry += E_[p, q] * h_[p, q]
    
    indices_two_electron_integrals = np.transpose((abs(g_)>cut_off_integral).nonzero())
    for indices in indices_two_electron_integrals:
        p, q, r, s = indices 
        e_pqrs = E_[p, q] @ E_[r, s] 
        if q == r: 
            e_pqrs += - E_[p, s]
        H_chemistry += e_pqrs * g_[p, q, r, s] / 2. 
        
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

    return H_chemistry


def build_hamiltonian_fermi_hubbard(h_,
                                    U_,
                                    nbody_basis,
                                    a_dagger_a,
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
    n_mo = np.shape(h_)[0]
    
    E_ = np.empty((n_mo, n_mo), dtype=object)
    for p in range(n_mo):
        for q in range(n_mo):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]
        
    # Building the N-electron Fermi-Hubbard matrix hamiltonian (Sparse)
    H_fermi_hubbard = scipy.sparse.csr_matrix((dim_H, dim_H))
    
    indices_one_electron_integrals = np.transpose((abs(h_)>cut_off_integral).nonzero())
    for indices in indices_one_electron_integrals:
        p, q = indices  
        H_fermi_hubbard += E_[p, q] * h_[p, q]
    
    indices_two_electron_integrals = np.transpose((abs(U_)>cut_off_integral).nonzero())
    for indices in indices_two_electron_integrals:
        p, q, r, s = indices  
        H_fermi_hubbard += a_dagger_a[2 * p, 2 * q] @ a_dagger_a[2 * r + 1, 2 * s + 1] * U_[p, q, r, s]
    
    if v_term is not None:
        indices_two_electron_integrals = np.transpose((abs(v_term)>cut_off_integral).nonzero())
        for indices in indices_two_electron_integrals:
            p, q, r, s = indices  
            H_fermi_hubbard +=  E_[p, q] @ E_[r, s] * v_term[p, q, r, s] 
            
    # global E_
    # E_ = np.empty((n_mo, n_mo), dtype=object)

    # # Building the N-electron Fermi-Hubbard matrix hamiltonian (Sparse)
    # H_fermi_hubbard = scipy.sparse.csr_matrix((dim_H, dim_H))
    # for p in range(n_mo):
    #     for q in range(n_mo):
    #         E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]
    #         H_fermi_hubbard += E_[p, q] * h_[p, q]
    #         for r in range(n_mo):
    #             for s in range(n_mo):
    #                 if U_[p, q, r, s] != 0:  # if U is 0, it doesn't make sense to multiply matrices
    #                     H_fermi_hubbard += a_dagger_a[2 * p, 2 * q] @ a_dagger_a[2 * r + 1, 2 * s + 1] * U_[p, q, r, s]

    # if v_term is not None:
    #     for p in range(n_mo):
    #         for q in range(n_mo):
    #             for r in range(n_mo):
    #                 for s in range(n_mo):
    #                     if v_term[p, q, r, s] != 0:  # if U is 0, it doesn't make sense to multiply matrices
    #                         H_fermi_hubbard += E_[p, q] @ E_[r, s] * v_term[p, q, r, s]

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

    return H_fermi_hubbard


def build_penalty_orbital_occupancy( a_dagger_a, occupancy_target):
    """
    Create a penalty operator to enforce a given occupancy number of electron
    in the molecular orbitals of the system

    Parameters
    ----------
    a_dagger_a :
        Matrix representation of the a_dagger_a operator
    occupancy_target : Iterable[int]
        Target number of electrons in a given molecular orbital

    Returns
    -------
    occupancy_penalty
        penalty operator
    list_good_states_indices
        list of many body state indices respecting the constraint
    list_bad_states_indices
        list of many body state indices not respecting the constraint

    Examples TODO: Saad help!
    ________
    >>> nbody_basis = build_nbody_basis(2, 2)
    >>> a_dagger_a = build_operator_a_dagger_a(nbody_basis)
    >>> build_penalty_orbital_occupancy(a_dagger_a, np.array([0.2, 0.8, 0, 0, 0 ,0]))
    (matrix([[ 3.6, -1.6,  0. ,  0. ,  0. ,  0. ],
             [-0.4,  0.4,  0. ,  0. ,  0. ,  0. ],
             [-0.4, -1.6,  2. ,  0. ,  0. ,  0. ],
             [-0.4, -1.6,  0. ,  2. ,  0. ,  0. ],
             [-0.4, -1.6,  0. ,  0. ,  2. ,  0. ],
             [-0.4, -1.6,  0. ,  0. ,  0. ,  4. ]]),
     array([], dtype=int64),
     array([0, 1, 2, 3, 4, 5], dtype=int64))

    """
    dim_H = a_dagger_a[0,0].shape[0]
    n_mo  = np.shape(a_dagger_a)[0] // 2
    occupancy_penalty = ( a_dagger_a[0, 0] + a_dagger_a[1, 1] -  occupancy_target * scipy.sparse.identity(dim_H) )**2
    for p in range(1,n_mo):
        occupancy_penalty  +=  ( a_dagger_a[2*p, 2*p] + a_dagger_a[2*p+1, 2*p+1] -  occupancy_target * scipy.sparse.identity(dim_H) )**2

    list_indices_good_states = np.where( np.diag( occupancy_penalty.A ) < 0.1 )[0]
    list_indices_bad_states  = np.where( np.diag( occupancy_penalty.A ) > 0.1 )[0]

    return  occupancy_penalty, list_indices_good_states, list_indices_bad_states


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


def build_full_mo_1rdm_and_2rdm_for_AS( WFT,
                                         a_dagger_a,
                                         frozen_indices,
                                         active_indices,
                                         n_mo_total ):
    """
    Create a full representation of the spin-free 1- and 2-electron reduced density matrices
    for a system with an active space. The wavefunction provided here describes ONLY the electronic
    strcuture within the active space.

    Parameters
    ----------
    WFT : array
        Reference wavefunction describing the electornic strcure in the active space ONLY.
    a_dagger_a : array
        Matrix representation of the a_dagger_a operator
    frozen_indices : array
        List of frozen indices
    active_indices : array
        List of active indices
    n_mo_total : int
        Total number of molecular orbitals

    Returns
    -------
    one_rdm : array
        1-electron reduced density matrix of the wavefunction
    two_rdm : array
        2-electron reduced density matrix of the wavefunction

    """
    if active_indices is not None:
        first_act_index = active_indices[0]
        n_active_mo = len( active_indices )

    one_rdm = np.zeros((n_mo_total, n_mo_total))
    two_rdm = np.zeros((n_mo_total, n_mo_total, n_mo_total, n_mo_total))

    global E_
    global e_

    E_ = np.empty((n_active_mo, n_active_mo), dtype=object)
    e_ = np.empty((n_active_mo, n_active_mo, n_active_mo, n_active_mo), dtype=object)

    if active_indices is not None:
        for p in range(n_active_mo):
            for q in range(n_active_mo):
                E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]

        for p in range(n_active_mo):
            for q in range(n_active_mo):
                for r in range(n_active_mo):
                    for s in range(n_active_mo):
                        e_[p, q, r, s] = E_[p, q] @ E_[r, s]
                        if q == r:
                            e_[p, q, r, s] += - E_[p, s]

    # Creating RDMs elements only within the frozen space
    if frozen_indices is not None:
        for i in frozen_indices:
            for j in frozen_indices:
                one_rdm[i, j] = 2. * delta(i, j)
                for k in frozen_indices:
                    for l in frozen_indices:
                        # State A
                        two_rdm[i, j, k, l] = 4. * delta(i, j) * delta(k, l) - 2. * delta(i, l) * delta(j, k)

    if active_indices is not None:
        # Creating RDMs elements in the the active/frozen spaces
        for p in active_indices:
            for q in active_indices:
                # Shifting the indices
                p_ = p - first_act_index
                q_ = q - first_act_index
                # 1-RDM elements only within the active space
                # State A
                one_rdm[p, q] = WFT.T @ E_[p_, q_] @ WFT

                # 2-RDM elements only within the active space
                for r in active_indices:
                    for s in active_indices:
                        # Shifting the indices
                        r_ = r - first_act_index
                        s_ = s - first_act_index
                        # State A
                        two_rdm[p, q, r, s] = WFT.T @ e_[p_, q_, r_, s_] @ WFT

                if frozen_indices is not None:
                    # 2-RDM elements between the active and frozen spaces
                    for i in frozen_indices:
                        for j in frozen_indices:
                            # State A
                            two_rdm[i, j, p, q] = two_rdm[p, q, i, j] = 2. * delta(i, j) * one_rdm[p, q]
                            two_rdm[p, i, j, q] = two_rdm[j, q, p, i] = - delta(i, j) * one_rdm[p, q]

    return one_rdm, two_rdm



def build_1rdm_alpha(WFT, a_dagger_a):
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


def build_1rdm_beta(WFT, a_dagger_a):
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


def build_1rdm_spin_free(WFT, a_dagger_a):
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


def build_2rdm_fh_on_site_repulsion(WFT, a_dagger_a, mask=None):
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


def build_2rdm_fh_dipolar_interactions(WFT, a_dagger_a, mask=None):
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


def build_2rdm_spin_free(WFT, a_dagger_a):
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


def build_1rdm_and_2rdm_spin_free(WFT, a_dagger_a):
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



def build_hybrid_1rdm_alpha_beta(WFT, a_dagger_a):
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



def build_transition_1rdm_alpha(WFT_A, WFT_B, a_dagger_a):
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



def build_transition_1rdm_beta(WFT_A, WFT_B, a_dagger_a):
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



def build_transition_1rdm_spin_free(WFT_A, WFT_B, a_dagger_a):
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


def build_transition_2rdm_spin_free(WFT_A, WFT_B, a_dagger_a):
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


# =============================================================================
#   FUNCTIONS TO CREAT PERSONALIZED MANY_BODY STATES AND PROJECTORS
# =============================================================================

def my_state( slater_determinant, nbody_basis ):
    """
    Translate a Slater determinant (occupation number list) into a many-body
    state referenced into a given Many-body basis.

    Parameters
    ----------
    slater_determinant  : array
        occupation number list
    nbody_basis         : array
        List of many-body states (occupation number states)

    Returns
    -------
    state :  array
        The slater determinant translated into the "kappa" many-body basis

    """
    kappa = np.flatnonzero((nbody_basis == slater_determinant).all(1))[0]  # nbody_basis.index(slater_determinant)
    state = np.zeros(np.shape(nbody_basis)[0])
    state[kappa] = 1.

    return state


def build_projector_active_space( n_elec,
                                  frozen_indices,
                                  active_indices,
                                  virtual_indices,
                                  nbody_basis,
                                  show_states=False ):
    """
    Build a many-body projector operator including all the many-body configurations
    respecting an active space structure such that :

                       | Phi ⟩ = | frozen, active, virtual ⟩

    Parameters
    ----------
    n_elec          : int,
        Total number of electron in the system
    frozen_indices  : array
        List of doubly occupied frozen orbitals
    active_indices  : array
        List of active orbitals
    virtual_indices : array
        List of virtual unoccupied orbitals
    nbody_basis     : array
        List of many-body states (occupation number states)

    Returns
    -------
    Proj_AS :  array
        Projector associated to the active-space defined

    """
    n_mo = len(frozen_indices) + len(active_indices) + len(virtual_indices)
    N_elec_frozen = 2 * len(frozen_indices)
    state_created_frozen = np.zeros(2*n_mo, dtype=np.int32)
    state_created_active = np.zeros(2*n_mo, dtype=np.int32)

    if show_states:
        print()
        print(' =>> List of states considered in the projector')

    for i in range(n_mo):
        if (i in frozen_indices):
            state_created_frozen[2*i]   = 1
            state_created_frozen[2*i+1] = 1
        elif (i in active_indices):
            state_created_active[2*i]   = 1
            state_created_active[2*i+1] = 1

    Proj_AS = np.zeros((len(nbody_basis),len(nbody_basis)))
    for state in nbody_basis:
        if (    state_created_frozen @ state == N_elec_frozen
            and state_created_active @ state == n_elec - N_elec_frozen ):

            fstate_created = my_state( state, nbody_basis )
            Proj_AS += np.outer(fstate_created, fstate_created)

            if show_states:
               print(state)

    return Proj_AS


# =============================================================================
#   INTEGRALS BUILDER FOR ACTIVE SPACE CALCULATION
# =============================================================================

def fh_get_active_space_integrals ( h_MO,
                                    U_MO,
                                    frozen_indices=None,
                                    active_indices=None ):
    """
    Restricts a Fermi-Hubbard system at a spatial orbital level to an active space.
    This active space may be defined by a list of active indices and
    doubly occupied indices. Note that one_body_integrals and
    two_body_integrals must be defined in an orthonormal basis set (MO like).

    Parameters
    ----------
    h_MO : array
        1 body integrals
    U_MO : array
        2 body integrals (coulomb interactions)
    frozen_indices: array
        A list of spatial orbital indices indicating which orbitals should be considered doubly occupied
    active_indices : array
        A list of spatial orbital indices indicating which orbitals should be considered active


    Returns
    -------
    core_energy : float
        Adjustment to constant shift in Hamiltonian from integrating out core orbitals
    h_act : array
        effective one-electron integrals over active space
    U_MO_active : array
        two-electron integrals over active space (coulomb interactions)

    """

    # Determine core constant ========
    # ==> From the U term
    core_energy = 0.0
    for i in frozen_indices:
        core_energy += 2 * h_MO[i, i]
        for j in frozen_indices:
            core_energy += U_MO[i, i, j, j]

    # Modified one electron integrals ========
    # ==> From the U term
    h_act = np.copy(h_MO)
    for t in active_indices:
        for u in active_indices:
            for i in frozen_indices:
                h_act[t, u] += U_MO[i, i, t, u]

    return (core_energy,
            h_act[np.ix_(active_indices, active_indices)],
            U_MO[np.ix_(active_indices, active_indices, active_indices, active_indices)] )


def fh_get_active_space_integrals_with_V( h_MO,
                                          U_MO,
                                          V_MO,
                                          frozen_indices=None,
                                          active_indices=None ):
    """
    Similar function as before but with an additional dipolar term V  when considering a Fermi-Hubbard system.
    Restricts the system at a spatial orbital level to an active space. 
    This active space may be defined by a list of active indices and
    doubly occupied indices. Note that one_body_integrals and
    two_body_integrals must be defined in an orthonormal basis set (MO like).

    Parameters
    ----------
    h_MO : array
        1 body integrals
    U_MO : array
        2 body elecronic integrals (coulomb interaction)
    V_MO : array
        2 body electronic integrals (dipolar interaction)
    frozen_indices: array
        A list of spatial orbital indices indicating which orbitals should be considered doubly occupied
    active_indices : array
        A list of spatial orbital indices indicating which orbitals should be considered active


    Returns
    -------
    core_energy : float
        Adjustment to constant shift in Hamiltonian from integrating out core orbitals
    h_act : array
        effective one-electron integrals over active space
    U_MO_active : array
        two-electron integrals over active space (coulomb interactions)
    V_MO_active : array
        two-electron integrals over active space (dipolar interactions)

    """

    # Determine core constant ========
    # ==> From the V term
    core_energy = 0.0
    for i in frozen_indices:
        for t in active_indices:
            core_energy += 2 * V_MO[i,t,t,i] # New contribution
        for j in frozen_indices:
                core_energy +=   4 * V_MO[i, i, j, j]

    # ==> From the U term
    for i in frozen_indices:
        core_energy += 2 * h_MO[i, i]
        for j in frozen_indices:
            core_energy += U_MO[i, i, j, j]

    # Modified one electron integrals ========
    # ==> From the V term
    h_act = h_MO.copy()
    for i in frozen_indices:
        for t in active_indices:
            for u in active_indices:
                h_act[t, u] +=  4 * V_MO[i, i, t, u]    -  V_MO[u, i, i, t]

    # ==> From the U term
    for t in active_indices:
        for u in active_indices:
            for i in frozen_indices:
                h_act[t, u] += U_MO[i, i, t, u]

    return (core_energy,
            h_act[np.ix_(active_indices, active_indices)],
            U_MO[np.ix_(active_indices, active_indices, active_indices, active_indices)],
            V_MO[np.ix_(active_indices, active_indices, active_indices, active_indices)] )



def qc_get_active_space_integrals(h_MO,
                                  g_MO,
                                  frozen_indices=None,
                                  active_indices=None):
    """
    Restricts an ab initio  electronic structure system at a spatial orbital level to an active space.
    This active space may be defined by a list of active indices and
    doubly occupied indices. Note that one_body_integrals and
    two_body_integrals must be defined in an orthonormal basis set (MO like).

    Parameters
    ----------
    h_MO : array
        1 body integrals
    g_MO : array
        2 body integrals
    frozen_indices: array
        A list of spatial orbital indices indicating which orbitals should be considered doubly occupied
    active_indices : array
        A list of spatial orbital indices indicating which orbitals should be considered active


    Returns
    -------
    core_energy : float
        Adjustment to constant shift in Hamiltonian from integrating out core orbitals
    h_act : array
        effective one-electron integrals over active space
    g_MO_active : array
        two-electron integrals over active space

    """
    # Fix data type for a few edge cases
    frozen_indices = [] if frozen_indices is None else frozen_indices
    if len(active_indices) < 1:
        raise ValueError('Some active indices required for reduction.')

    # Determine core constant
    core_energy = 0.0
    for i in frozen_indices:
        core_energy += 2 * h_MO[i, i]
        for j in frozen_indices:
            core_energy += (2 * g_MO[i, i, j, j]
                              - g_MO[i, j, j, i])
            # core_constant += (2 * two_body_integrals[i, j, i, j]
            #                   - two_body_integrals[i, j, j, i])

    # Modified one electron integrals
    h_act = np.copy(h_MO)
    for u in active_indices:
        for v in active_indices:
            for i in frozen_indices:
                h_act[u, v] += ( 2 * g_MO[i, i, u, v]  - g_MO[i, u, v, i] )

    # Restrict integral ranges and change M appropriately
    return (core_energy,
            h_act[np.ix_(active_indices, active_indices)],
            g_MO[np.ix_(active_indices, active_indices, active_indices, active_indices)])



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



def build_local_s2_sz_splus_operator( a_dagger_a, list_mo_local ):
    """
    Create a matrix representation of the spin operators s2, s_z and s_plus 
    in the many-body basis projected into a local basis defined by list_mo_local. 

    Parameters
    ----------
    a_dagger_a : array
        a_dagger_a operator.
    list_mo_local : list
        indices of the local Molecular Orbitals.

    Returns
    -------
    s2_local : array
        matrix representation of the s2_local operator (local spin squared) in the many-body basis.
    s_z_local : array
        matrix representation of the s_z_local operator (local spin projection) in the many-body basis.
    s_plus_local : array
        matrix representation of the s_plus_local operator (local spin ladder up) in the many-body basis.

    """
    dim_H = np.shape(a_dagger_a[0, 0].A)[0]
    s_plus_local = scipy.sparse.csr_matrix((dim_H, dim_H))
    s_z_local = scipy.sparse.csr_matrix((dim_H, dim_H))
    
    for p in list_mo_local:
        s_plus_local += a_dagger_a[2 * p, 2 * p + 1]
        s_z_local += (a_dagger_a[2 * p, 2 * p] - a_dagger_a[2 * p + 1, 2 * p + 1]) / 2.
    s2_local = s_plus_local @ s_plus_local.T + s_z_local @ s_z_local - s_z_local

    return s2_local, s_z_local, s_plus_local


def build_sAsB_coupling( a_dagger_a, list_mo_local_A, list_mo_local_B ):
    """
    Create a matrix representation of the product of two local spin operators
    s_A x s_B in the many-body basis. Each one being associated to a local set
    of molecular orbitals attached to two different "molecular fragments" A and B.

    Parameters
    ----------
    a_dagger_a : array
        matrix representation of the a_dagger_a operator in the many-body basis.
    list_mo_local_A : array
        List of molecular orbital indices belonging to fragment A
    list_mo_local_B : array
        List of molecular orbital indices belonging to fragment B

    Returns
    -------
    sAsB_coupling : array
        matrix representation of s_A x s_B in the many-body basis.

    """
    dim_H = np.shape(a_dagger_a[0, 0].A)[0]
    s_plus_A = scipy.sparse.csr_matrix((dim_H, dim_H))
    s_plus_B = scipy.sparse.csr_matrix((dim_H, dim_H))
    s_z_A = scipy.sparse.csr_matrix((dim_H, dim_H))
    s_z_B = scipy.sparse.csr_matrix((dim_H, dim_H))
    for p in list_mo_local_A:
        s_plus_A += a_dagger_a[2 * p, 2 * p + 1]
        s_z_A += (a_dagger_a[2 * p, 2 * p] - a_dagger_a[2 * p + 1, 2 * p + 1]) / 2.
    for p in list_mo_local_B:
        s_plus_B += a_dagger_a[2 * p, 2 * p + 1]
        s_z_B += (a_dagger_a[2 * p, 2 * p] - a_dagger_a[2 * p + 1, 2 * p + 1]) / 2.

    sAsB_coupling = 0.5 * ( s_plus_A @ s_plus_B.T + s_plus_A.T @ s_plus_B ) + s_z_A @ s_z_B

    return sAsB_coupling


def build_spin_subspaces( S2_local, S2_local_target ):
    """

    Create a projector over the many-body space spanning all the configurations
    which should be counted to produce a local spin s2_local of value given by
    s2_local_target.

    Parameters
    ----------
    s2_local        : array
        Local spin operator associated to a restricted number of orbitals
    s2_local_target :array
        Value of the local spin target to create the assoacited many-body subspace

    Returns
    -------
    Projector_spin_subspace : array
        Projector over the many-body sub-space targeted (*i.e.* outer product of
        the many-body states respecting the local spin symmetry demanded)

    """
    S2_local_eigval, S2_local_eigvec = scipy.linalg.eigh( S2_local.A )
    Set_vectors = S2_local_eigvec[ :,   (S2_local_eigval >= S2_local_target-1.e-3)
                                      & (S2_local_eigval <= S2_local_target+1.e-3)   ]
    Projector_spin_subspace  =  Set_vectors @ Set_vectors.T

    return Projector_spin_subspace

# =============================================================================
#  SPIN-ORBIT OPERATORS
# =============================================================================

def build_local_l2_lz_lplus_operator(a_dagger_a, list_mo_local, l_local, list_l_local):
    """
    Create a matrix representation of the angular momentum operators l2, l_z and l_plus
    in the many-body basis projected into a local basis defined by list_mo_local. 

    Parameters
    ----------
    a_dagger_a : array
        a_dagger_a operator.
    list_mo_local : list
        indices of the local Molecular Orbitals.
    l_local : int
        azimutal quantum number of the local MOs (1: p, 2: d, 3: f, 4: g)
    list_l_local : list
        local angular momentum associated to each local MO.

    Returns
    -------
    l2_local : array
        matrix representation of the l2_local operator (local angular momentum squared) in the many-body basis.
    l_z_local : array
        matrix representation of the l_z_local operator (local angular momentum projection) in the many-body basis.
    l_plus_local : array
        matrix representation of the l_plus_local operator (local angular momentum ladder up) in the many-body basis.

    """
    dim_H = np.shape(a_dagger_a[0, 0].A)[0]
    l_plus_local = scipy.sparse.csr_matrix((dim_H, dim_H))
    
    l_z_local = scipy.sparse.csr_matrix((dim_H, dim_H))
    
    for m in list_mo_local: # -l_loca < m < +l_local
        for m2 in list_mo_local:
            if m2 == m:
                l_z_local += (list_l_local[m]) *( a_dagger_a[2 * m, 2 * m] + a_dagger_a[2 * m + 1, 2 * m + 1])  
            elif m2 != m:
                if np.abs(list_l_local[m] - list_l_local[m2]) != 1:
                    pass
                # Part for l_plus
                elif list_l_local[m2] - list_l_local[m] == 1:
                    l_plus_local += np.sqrt(l_local+list_l_local[m]+1) * np.sqrt(l_local - list_l_local[m]) * (a_dagger_a[2 * m2, 2 * m]
                                                                                           + a_dagger_a[2*m2+1, 2*m+1])       
            
    l2_local = l_plus_local @ l_plus_local.T + l_z_local @ l_z_local - l_z_local
    
    return l2_local, l_z_local, l_plus_local

def build_local_j2_jz_jplus_operator(a_dagger_a, list_mo_local, l_local, list_l_local):
    """
    Create a matrix representation of the total angular momentum operators j2, j_z and j_plus
    in the many-body basis projected into a local basis defined by list_mo_local. 

    Parameters
    ----------
    a_dagger_a : array
        a_dagger_a operator.
    list_mo_local : list
        indices of the local Molecular Orbitals.
    l_local : int
        azimutal quantum number of the local MOs (1: p, 2: d, 3: f, 4: g)
    list_l_local : list
        local angular momentum associated to each local MO.

    Returns
    -------
    j2_local : array
        matrix representation of the j2_local operator (local total angular momentum squared) in the many-body basis.
    j_z_local : array
        matrix representation of the j_z_local operator (local total angular momentum projection) in the many-body basis.
    j_plus_local : array
        matrix representation of the j_plus_local operator (local total angular momentum ladder up) in the many-body basis.

    """
    s2_local, s_z_local, s_plus_local = build_local_s2_sz_splus_operator( a_dagger_a, list_mo_local )
    l2_local, l_z_local, l_plus_local = build_local_l2_lz_lplus_operator(a_dagger_a, list_mo_local,l_local, list_l_local)
    
    j_plus_local = l_plus_local + s_plus_local
    j_z_local = l_z_local + s_z_local
    
    j2_local = j_plus_local @ j_plus_local.T + j_z_local @ j_z_local - j_z_local
    
    return j2_local, j_z_local, j_plus_local


def build_local_spinorbit_lz(a_dagger_a, list_mo_local, l_local, list_l_local):
    """
    Create a matrix representation of the spin-orbit operator
    in the many-body basis projected into a local basis defined by list_mo_local. 
    It is defined using projection and ladder operators of spin "S" and angular momentum "L".
    
    H_SO = L_z @ S_z + 1/2 * (L_p @ S_p.T + L_p.T @ S_p)
    
    This is equivalent to build_local_spinorbit_j2().

    Parameters
    ----------
    a_dagger_a : array
        a_dagger_a operator.
    list_mo_local : list
        indices of the local Molecular Orbitals.
    l_local : int
        azimutal quantum number of the local MOs (1: p, 2: d, 3: f, 4: g)
    list_l_local : list
        local angular momentum associated to each local MO.

    Returns
    -------
    H_SO : array
        matrix representation of the local H_SO operator (local spin-orbit) in the many-body basis.

    """
    ### General formula
    s2_local, s_z_local, s_plus_local = build_local_s2_sz_splus_operator( a_dagger_a, list_mo_local )
    l2_local, l_z_local, l_plus_local = build_local_l2_lz_lplus_operator(a_dagger_a, list_mo_local,l_local, list_l_local)
    
    H_SO = l_z_local @ s_z_local + (l_plus_local @ s_plus_local.T + l_plus_local.T @ s_plus_local) * 1/2
    
    return H_SO


def build_local_spinorbit_j2(a_dagger_a, list_mo_local, l_local, list_l_local):
    """
    Create a matrix representation of the spin-orbit operator
    in the many-body basis projected into a local basis defined by list_mo_local. 
    It is defined using the squared operator of spin "S", angular momentum "L" and total angular momentum "J".
    
    L.S = 1/2 * (J^2 - S^2 - L^2)
    
    This is equivalent to build_local_spinorbit_lz().
    
    Parameters
    ----------
    a_dagger_a : array
        a_dagger_a operator.
    list_mo_local : list
        indices of the local Molecular Orbitals.
    l_local : int
        azimutal quantum number of the local MOs (1: p, 2: d, 3: f, 4: g)
    list_l_local : list
        local angular momentum associated to each local MO.

    Returns
    -------
    LS : array
        matrix representation of the local LS operator (local spin-orbit) in the many-body basis.

    """
    s_2_local, s_z_local, s_plus_local = build_local_s2_sz_splus_operator( a_dagger_a, list_mo_local )
    l_2_local, l_z_local, l_plus_local = build_local_l2_lz_lplus_operator(a_dagger_a, list_mo_local,l_local, list_l_local)
    j_2_local, j_z_local, j_plus_local = build_local_j2_jz_jplus_operator(a_dagger_a, list_mo_local,l_local, list_l_local)

    LS = 1/2 * (j_2_local - s_2_local - l_2_local)

    return LS


# =============================================================================
#  FUNCTION TO GIVE ACCESS TO BASIC QUANTUM CHEMISTRY DATA FROM PSI4
# =============================================================================

def get_info_from_psi4( string_geometry,
                        basisset,
                        molecular_charge=0,
                        TELL_ME=False ):
    """
    Simple Psi4 interface to obtain relevant information for a quantum chemistry problem.
    Function to realise an Hartree-Fock calculation on a given molecule and to
    return all the associated information for further correlated wavefunction
    calculation for QuantNBody.

    Parameters
    ----------
    string_geometry  : str
        XYZ file for the calculation
    basisset         : str
        name of the basis we want to use
    molecular_charge : int
        value of the charge we want to attribute to the molecule (default is 0)
    TELL_ME          : Boolean
        In case we want the output (default is False)

    Returns
    -------
    overlap_AO  : array
        Overlap matrix in the AO basis
    h_AO        : array
        one-body integrals in the AO basis
    g_AO        : array
        two-body integrals in the AO basis
    C_RHF       : array
        HF Molecular orbital coefficient matrix
    E_HF        : float
        HF energy
    E_rep_nuc   : float
        Energy of the nuclei repulsion

    """
    if not TELL_ME:
        # To prevent psi4 from printing the output in the terminal
        psi4.core.set_output_file("output_Psi4.txt", False)

    # Clean all previous options for psi4
    psi4.core.clean()
    psi4.core.clean_variables()
    psi4.core.clean_options()

    # Adding a 'no symmetry' info to the geometry string
    string_geometry += '\n' + 'symmetry c1'

    # Adding a 'no symmetry' info to the geometry string
    molecule = psi4.geometry( string_geometry  )

    # Setting the charge of the moelcule
    molecule.set_molecular_charge( molecular_charge )

    # Setting the bassiset for the calculation
    psi4.set_options({'basis' : basisset})

    # Realizing a generic HF calculation ======
    E_HF, scf_wfn = psi4.energy( 'scf', molecule=molecule, return_wfn=True )

    # Nuclear repulsion energy
    E_rep_nuc = molecule.nuclear_repulsion_energy()

    # MO coeff matrix from the initial RHF calculation
    C_RHF     = np.asarray(scf_wfn.Ca()).copy()

    # Get AOs integrals using MintsHelper
    mints     = psi4.core.MintsHelper(scf_wfn.basisset())

    # Storing the AO overlap matrix
    overlap_AO = np.asarray(mints.ao_overlap())

    # 1-electron integrals in the original AO basis
    h_AO   = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

    # 2-electron integrals in the original AO basis
    g_AO   = np.asarray(mints.ao_eri()).reshape(( np.shape( h_AO )[0],
                                                 np.shape( h_AO )[0],
                                                 np.shape( h_AO )[0],
                                                 np.shape( h_AO )[0] ))

    return overlap_AO, h_AO, g_AO, C_RHF, E_HF, E_rep_nuc


# =============================================================================
#  FUNCTION TO EXPRESS CORRELATED WAVE FUNCTIONS
#  IN DIFFERENT MOLECULAR ORBITAL BASIS
# =============================================================================

@njit
def weight_det( C_B2_B1, occ_spinorb_Det1, occ_spinorb_Det2 ):
    """
    Evaluate the overlap of two slater determinants expressed in two different
    orbital basis. This is actually given by the determinant of the overlap of the
    occupied spin orbital present in each slater determinant.

    Parameters
    ----------
    C_B2_B1          : array
        Coefficient matrix of the MO basis 1 expressed in the MO basis 2
    occ_spinorb_Det1 : array
        Occupied spinorbital in the slater determinant 1 (bra)
    occ_spinorb_Det2 : array
        occupied spinorbital in the slater determinant 2 (ket)

    Returns
    -------
    Det_amplitude : float
        resulting determinant of the occupied spinorbital from the two different basis

    """
    # Number of electron = number of occupied spin-orbitals
    n_elec = len(occ_spinorb_Det1)
    # Matrix containing the final spin-orbital overlaps
    M      = np.zeros(( n_elec, n_elec ))

    ind_row = 0
    for i in occ_spinorb_Det1:
        ind_col = 0
        for j in occ_spinorb_Det2:
            if ( i%2==0 and j%2==0 ):
                M[ ind_row, ind_col ] = C_B2_B1[ i//2, j//2 ]
            elif ( (i+1)%2==0 and (j+1)%2==0 ) :
                M[ ind_row, ind_col ] = C_B2_B1[ i//2, j//2 ]
            ind_col += 1
        ind_row += 1
    Det_amplitude = np.linalg.det( M )
    return Det_amplitude


def scalar_product_different_MO_basis( Psi_A_MOB1,
                                       Psi_B_MOB2,
                                       C_MOB1,
                                       C_MOB2,
                                       nbody_basis ):
    """
    Evaluate the non-trivial scalar product of two multi-configurational
    wavefunction expressed in two different molecular orbital basis.

    Parameters
    ----------
    Psi_A_MOB1  : array
        Wavefunction A (will be a Bra) expressed in the first orbital basis
    Psi_B_MOB2  : array
        Wavefunction B (will be a Ket) expressed in the second orbital basis
    C_MOB1      : array
        First basis' Molecular orbital coefficient matrix
    C_MOB2      : array
        Second basis' Molecular orbital coefficient matrix
    nbody_basis : array
        List of many-body state

    Returns
    -------
    scalar_product : float
        Amplitude of the scalar product

    """
    dim_H = len(nbody_basis)

    # Overlap matrix in the common basis
    S = np.linalg.inv( C_MOB1 @ C_MOB1.T )
    # Building the matrix expressing the MO from B1 in the B2 basis
    C_B2_B1 = C_MOB1.T @ S @ C_MOB2

    scalar_product = 0
    for I in range(dim_H):
        if abs(Psi_A_MOB1[I]) > 1e-8:
            # Finding the indices of the spinorbitals which are occupied in the Det_I
            occ_spinorb_Det_I = np.nonzero( nbody_basis[I] )[0].tolist()
            for J in range(dim_H):
                if abs(Psi_B_MOB2[J]) > 1e-8:
                    # Finding the indices of the spinorbitals which are occupied in the Det_J
                    occ_spinorb_Det_J = np.nonzero( nbody_basis[J] )[0].tolist()
                    D  = weight_det( C_B2_B1, occ_spinorb_Det_I, occ_spinorb_Det_J )
                    scalar_product += D * np.conj(Psi_A_MOB1[I]) * Psi_B_MOB2[J]

    return scalar_product



def transform_psi_MO_basis1_in_MO_basis2( Psi_A_MOB1,
                                          C_MOB1,
                                          C_MOB2,
                                          nbody_basis ):
    """
    Transform a multi-configurational wavefunction originaly expressed in a
    molecular orbital basis B1 into another molecular orbital basis B2.
    Both basis are described respectively by two MO coefficient matrix which have
    to express their respetive basis MOs in a same common basis
    (e.g. AO or any other MO basis).

    Parameters
    ----------
    Psi_A_MOB1  : array
        Wavefunction A (will be a Bra) expressed in the first orbital basis
    C_MOB1      : array
        First basis' Molecular orbital coefficient matrix
    C_MOB2      : array
        Second basis' Molecular orbital coefficient matrix
    nbody_basis : array
        List of many-body state

    Returns
    -------
    Psi_A_MOB2 :  array
        Final shape of the multi-configurational wavefunction in the second MO basis B2

    """
    dim_H = len(nbody_basis)

    # Overlap matrix in the common basis
    S = scipy.linalg.inv( C_MOB1 @ C_MOB1.T )
    # Building the matrix expressing the MO from B1 in the B2 basis
    C_B2_B1 = C_MOB1.T @ S @ C_MOB2

    Psi_A_MOB2 = np.zeros_like(Psi_A_MOB1)
    for I in range(dim_H):
        if abs(Psi_A_MOB1[I]) > 1e-8:
            # Finding the indices of the spinorbitals which are occupied in the Det_I
            occ_spinorb_Det_I = np.nonzero( nbody_basis[I] )[0].tolist()
            for J in range(dim_H):
                # Finding the indices of the spinorbitals which are occupied in the Det_J
                occ_spinorb_Det_J = np.nonzero( nbody_basis[J] )[0].tolist()
                D  = weight_det( C_B2_B1, occ_spinorb_Det_I, occ_spinorb_Det_J )
                Psi_A_MOB2[J] += D *  Psi_A_MOB1[I]

    return Psi_A_MOB2


# def TEST_transform_psi_MO_basis1_in_MO_basis2( Psi_A_MOB1,
#                                               C_MOB1,
#                                               C_MOB2,
#                                               nbody_basis,
#                                               frozen_indices=None):
#     """
#     Transform an intial multi-configurational wavefunction expressed with
#     an intial molecular orbital basis 1 into another molecular orbital basis 2

#     Parameters
#     ----------
#     Psi_A_MOB1  : array
#         Wavefunction A (will be a Bra) expressed in the first orbital basis
#     C_MOB1      : array
#         First basis' Molecular orbital coefficient matrix
#     C_MOB2      : array
#         Second basis' Molecular orbital coefficient matrix
#     nbody_basis : array
#         List of many-body state

#     Returns
#     -------
#     Psi_A_MOB2 :  array
#         Final shape of the cmulti-onfigurational wft in the second MO basis

#     """
#     dim_H = len(nbody_basis)

#     # Overlap matrix in the common basis
#     S = scipy.linalg.inv( C_MOB1 @ C_MOB1.T )
#     # Building the matrix expressing the MO from B1 in the B2 basis
#     C_B2_B1 = C_MOB1.T @ S @ C_MOB2

#     frozen_spinorb_indices = []
#     if frozen_indices != None:
#         frozen_spinorb_indices = [ 1 for i in range(2*len(frozen_indices)) ]

#     Psi_A_MOB2 = np.zeros_like(Psi_A_MOB1)
#     for I in range(dim_H):
#         if abs(Psi_A_MOB1[I]) > 1e-8:
#             # Finding the indices of the spinorbitals which are occupied in the Det_I
#             occ_spinorb_Det_I = np.nonzero( frozen_spinorb_indices +  list(nbody_basis[I]) )[0].tolist()
#             for J in range(dim_H):
#                 # Finding the indices of the spinorbitals which are occupied in the Det_J
#                 occ_spinorb_Det_J = np.nonzero( frozen_spinorb_indices + list(nbody_basis[J]) )[0].tolist()
#                 D  = weight_det( C_B2_B1, occ_spinorb_Det_I, occ_spinorb_Det_J )
#                 Psi_A_MOB2[J] += D *  Psi_A_MOB1[I]

#     return Psi_A_MOB2


def scalar_product_different_MO_basis_with_frozen_orbitals( Psi_A_MOB1,
                                                            Psi_B_MOB2,
                                                            C_MOB1,
                                                            C_MOB2,
                                                            nbody_basis,
                                                            frozen_indices=None):
    """
    Evaluate the non-trivial scalar product of two multi-configurational
    wavefunction expressed in two different moelcular orbital basis. Each
    of these wavefunction has a same number of doubly occupied (frozen) orbital.

    Parameters
    ----------
    Psi_A_MOB1  : Wavefunction A (will be a Bra) expressed in the first orbital basis
    Psi_B_MOB2  : Wavefunction B (will be a Ket) expressed in the second orbital basis
    C_MOB1      : First basis' Molecular orbital coefficient matrix
    C_MOB2      : Second basis' Molecular orbital coefficient matrix
    nbody_basis : List of many-body state

    Returns
    -------
    scalar_product :  Amplitude of the scalar product

    """
    dim_H = len( nbody_basis )

    # Overlap matrix in the common basis
    S = scipy.linalg.inv( C_MOB1 @ C_MOB1.T )
    # Building the matrix expressing the MO from B1 in the B2 basis
    C_B2_B1 = C_MOB1.T @ S @ C_MOB2

    frozen_spinorb_indices = []
    if frozen_indices != None:
        frozen_spinorb_indices = [ 1 for i in range(2*len(frozen_indices)) ]

    scalar_product = 0
    for I in range(dim_H):
        if abs(Psi_A_MOB1[I]) > 1e-8:
            # Finding the indices of the spinorbitals which are occupied in the Det_I
            occ_spinorb_Det_I = np.nonzero( frozen_spinorb_indices +  list(nbody_basis[I]) )[0].tolist()

            for J in range(dim_H):
                if abs(Psi_B_MOB2[J]) > 1e-8:
                    # Finding the indices of the spinorbitals which are occupied in the Det_J
                    occ_spinorb_Det_J = np.nonzero( frozen_spinorb_indices +  list(nbody_basis[J]) )[0].tolist()
                    D  = weight_det( C_B2_B1, occ_spinorb_Det_I, occ_spinorb_Det_J )
                    scalar_product += D * np.conj(Psi_A_MOB1[I]) * Psi_B_MOB2[J]

    return scalar_product

# =============================================================================
#  FUNCTION TO HELP THE VISUALIZATION OF MANY-BODY WAVE FUNCTIONS
# =============================================================================

def visualize_wft(WFT, nbody_basis, cutoff=0.005, ndets=8, atomic_orbitals=False):
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
    ndets            : int
        Maximum number of printed determinants
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
    for index in list_sorted_index[0:ndets]:
        state = states[index]
        True_index_state =  np.flatnonzero((nbody_basis == state).all(1))[0]
        if atomic_orbitals:
            ket = get_ket_in_atomic_orbitals(state, bra=False)
        else:
            ket = '|' + "".join([str(elem) for elem in state]) + '⟩'
        return_string += f'\t{coefficients[index]:+1.5f}   {ket}    #{True_index_state} \n'
    print(return_string)
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
    ax.bar_label(rects1, labels=[
        rf'$| {" ".join(map(str, nbody_basis[j]))}\rangle$' for j in range(len(coefficients))],
        fontsize=size_label, padding=2)

    # Add grid to the x-axis for better readability
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)

    return


def plot_wavefunctions(WFT, nbody_basis, list_states=[0], probability=False, cutoff=0.005, label_props=["x-large", 16, 16, 16]):
    """
    Plot the wavefunctions chosen by the user.

    Parameters
    ----------
    WFT : numpy array
        A 2D array containing the wavefunctions.
    nbody_basis : list
        A list of the many-body basis states.
    list_states : list
        A list of the states to plot. Default is [0] the ground state
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
    n_states = len(list_states)  # Number of states to plot
    max_abs_coefficient = np.max(np.abs(WFT[:, list_states]))
    damping = 0.25  # Damping factor for x-ticks

    # Create a grid of subplots with shared x-axis
    list_states = list_states[::-1]
    indices_by_state = [np.where(np.abs(WFT[:, i]) > cutoff)[
        0] for i in list_states]
    heights = [len(indices) for indices in indices_by_state]
    gridspecs = {'height_ratios': heights} if n_states > 1 else {
        'height_ratios': None}

    fig, axs = plt.subplots(n_states, 1, sharex=True,
                            dpi=150, gridspec_kw=gridspecs)

    # Loop through each state to plot
    for i, state_index in enumerate(list_states):
        ax = axs[i] if n_states > 1 else axs
        indices = indices_by_state[i]
        # Get indices where the coefficients exceed the cutoff value

        # Collect coefficients and corresponding many-body states
        coefficients = WFT[indices,
                           state_index]**2 if probability else WFT[indices, state_index]
        states = [nbody_basis[j] for j in indices]

        # Sort coefficients and states by the absolute value of the coefficients
        combined = list(zip(np.abs(coefficients), coefficients, states))
        sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
        sorted_abs_coefficients, sorted_coefficients, sorted_nbody_basis = zip(
            *sorted_combined)

        bar_y(ax, sorted_coefficients, sorted_nbody_basis, label_props[0])

        # Remove y-ticks and plot the bars
        ax.set_yticks([])
        ax.text(-0.07, 0.5, rf'$|\Psi_{{{state_index}}}\rangle$', ha='center', va='center',
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
    end_tick = max_abs_coefficient**2 + \
        damping if probability else max_abs_coefficient + damping
    x_ticks = np.arange(0., end_tick, 0.1)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=label_props[3])
    # Show the plot

    plt.tight_layout()
    plt.show()
    return



# =============================================================================
# ORBITAL OPTIMIZATION
# =============================================================================

# def prepare_vector_k_orbital_rotation_with_active_space( Vec_k,
#                                                          n_mo,
#                                                          frozen_indices,
#                                                          active_indices,
#                                                          virtual_indices):
#     """
#     Create an initial

#     Parameters
#     ----------
#     Vec_k           :
#     n_mo            :
#     frozen_indices  :
#     active_indices  :
#     virtual_indices :

#     Returns
#     -------
#     Vec_k :

#     """
#     Vec_k = []
#     for p in range(n_mo-1):
#         for q in range(p+1, n_mo):
#             if not( ( p in active_indices and q in active_indices)
#                  or ( p in frozen_indices and q in frozen_indices)
#                  or ( p in virtual_indices and q in virtual_indices)):
#                 Vec_k += [ 0 ]

#     return Vec_k

# def transform_vec_to_skewmatrix_with_active_space( Vec_k,
#                                                    n_mo,
#                                                    frozen_indices,
#                                                    active_indices ):
#     """
#     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     Function to build the skew-matrix (Anti-Symmetric) generator matrix K for
#     the orbital rotations from a vector k.
#     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     """
#     Skew_Matrix_K = np.zeros((n_mo, n_mo))
#     ind_ij = 0
#     for j in range(n_mo-1):
#         for i in range(j+1, n_mo):
#             Skew_Matrix_K[i, j] = Vec_k[ind_ij]
#             Skew_Matrix_K[j, i] = - Vec_k[ind_ij]
#             ind_ij += 1
#     return Skew_Matrix_K


@njit
def compute_energy_with_rdm(ONE_RDM,
                            TWO_RDM,
                            active_indices,
                            h_MO,
                            g_MO,
                            E_shift):
    """
    Compute the energy based on the RDMs, the integral and the energy shift (E_CoreMF + E_rep_nuc)

    Parameters
    ----------
    ONE_RDM         : 1-electron reduced density matrix
    TWO_RDM         : 2-electron reduced density matrix
    active_indices  : list of avtice indices
    h_MO            : 1-electron integrals
    g_MO            : 2-electron integrals
    E_shift         : Energy shift = MeanField energy (occupied space) + Nuclei repulsion energy

    Returns
    -------
    energy          : Final energy of the system

    """
    max_ = active_indices[-1] + 1
    energy = E_shift
    for p in range(max_):
        for q in range(max_):
            energy += ONE_RDM[p, q] * h_MO[p, q]
            for r in range(max_):
                for s in range(max_):
                    energy += 0.5 * TWO_RDM[p, q, r, s] * g_MO[p, q, r, s]
    return energy



def prepare_vector_k_orbital_rotation_with_active_space(n_mo,
                                                         frozen_indices,
                                                         active_indices,
                                                         virtual_indices):
    """
    Prepare the initial vector of kappa parameters (size and amplitude)
    for orbital optimization. The initialization is realized with zeros.

    Parameters
    ----------
    n_mo            : int
        Number of orbital
    frozen_indices  : array 
        list of frozen indices
    active_indices  : array
        list of active indices
    virtual_indices : array
        list of virtual indices

    Returns
    -------
    Vec_k :  array
        final vector prepared with zeros 

    """
    Vec_k = []
    for p in range(n_mo-1):
        for q in range(p+1, n_mo):
            if not( ( p in frozen_indices and q in frozen_indices)
                 or ( p in virtual_indices and q in virtual_indices)):
                Vec_k += [ 0. ]

    return Vec_k


def transform_vec_to_skewmatrix_with_active_space( Vec_k,
                                                   n_mo,
                                                   frozen_indices,
                                                   active_indices,
                                                   virtual_indices):
    """
    Build from a vector containing the rotation parameters the skew-matrix
    (Anti-Symmetric) generator matrix K for the orbital rotations genetor exp(K).

    Parameters
    ----------
    Vec_k           :  array 
        vector containing the kappa amplitude
    n_mo            :  int 
        number of orbital
    frozen_indices  :  array 
        list of frozen indices
    active_indices  :  array 
        list of active indices
    virtual_indices :  array 
        list of virtual indices

    Returns
    -------
    Skew_Matrix_K   :  array
        Final matrix K to be exponentiated

    """
    Skew_Matrix_K = np.zeros((n_mo, n_mo))
    ind_pq = 0
    for q in range(n_mo-1):
        for p in range(q+1, n_mo):
            if not(  ( p in frozen_indices and q in frozen_indices)
                  or ( p in virtual_indices and q in virtual_indices)):
                Skew_Matrix_K[p, q] = Vec_k[ind_pq]
                Skew_Matrix_K[q, p] = - Vec_k[ind_pq]
                ind_pq += 1
    return Skew_Matrix_K



def energy_cost_function_orbital_optimization( Vec_k,
                                                one_rdm,
                                                two_rdm,
                                                h,
                                                g,
                                                E_rep_nuc,
                                                frozen_indices,
                                                active_indices,
                                                virtual_indices ):
    """
    Energy cost function for a brute force orbital optimization with scipy
    (based on spin-free RDMs and ab initio problem)

    Parameters
    ----------
    Vec_k           :  array
        vector containing all the orbital rotation parameters
    one_rdm         :  array
        1-electron reduced density matrix
    two_rdm         :  array
        2-electron reduced density matrix
    h               :  array
        1-electron integral to be transformed by orbital rotation
    g               :  array
        2-electron integral to be transformed by orbital rotation
    E_rep_nuc       :  float
        Energy of repulsion between the nuclei
    frozen_indices  :  array
        list of frozen indices
    active_indices  :  array
        list of active indices
    virtual_indices :  array
        list of virtual indices

    Returns
    -------
    E_new :  float
        Final energy after playing with the orbital rotation parameters

    """
    n_mo = np.shape(h)[0]
    K_mat = transform_vec_to_skewmatrix_with_active_space(Vec_k,
                                                          n_mo,
                                                          frozen_indices,
                                                          active_indices,
                                                          virtual_indices)
    U_OO =  scipy.linalg.expm( - K_mat )
    h_new, g_new = transform_1_2_body_tensors_in_new_basis( h, g, U_OO )

    E_new = compute_energy_with_rdm( one_rdm,
                                    two_rdm,
                                    active_indices,
                                    h_new,
                                    g_new,
                                    E_rep_nuc )
    return E_new



def brute_force_orbital_optimization( one_rdm,
                                      two_rdm,
                                      h,
                                      g,
                                      E_rep_nuc,
                                      C_ref,
                                      frozen_indices,
                                      active_indices,
                                      virtual_indices,
                                      max_iteration=1000,
                                      method_name='BFGS',
                                      grad_tolerance=1e-6,
                                      show_me=False,
                                      SAD_guess = False):
    """
    Method implementing a brute force orbtial optimization using scipy optimizer
    (based on spin-free RDMs and ab initio problem)

    Parameters
    ----------
    one_rdm         :  array
        1-electron reduced density matrix
    two_rdm         :  array
        2-electron reduced density matrix
    h               :  array
        1-electron integral to be transformed
    g               : array
        2-electron integral to be transformed
    E_rep_nuc       :  float
        Energy of repulsion between the nuclei
    C_ref           :  array
        Inital coefficient matrix for the molecular orbital (to be rotated)
    frozen_indices  : array
        List of frozen indices
    active_indices  :  array
        List of active indices
    virtual_indices :  array
        List of virtual indices
    max_iteration   :  int
        Maximum number of iteration for the optimization. The default is 1000.
    method_name     :  str
        Method name for the orbital optimization. (The default is 'BFGS')
    grad_tolerance  :  float
        Gradient threshold for the convergence of the optmization. The default is 1e-6.
    show_me         :  boolean
        To show the evolution of the optimizaiton process. The default is False.
    SAD_guess       :  boolean
        Guess orbital (important for HF orbital optimization) implementing a
        " Superposition of Atomic Density"(SAD guess). (The default is False.)

    Returns
    -------
    C_OO    :  array
        Orbital-optimized molecular orbital coefficient matrix
    E_new   :  float
        Final energy after orbital optimizaiton
    h_OO    :  array
        Orbital-optimized 1-electron integrals
    g_OO    :  array
        Orbital-optimized 2-electron integrals

    """
    n_mo = np.shape(h)[0]
    Vec_k = prepare_vector_k_orbital_rotation_with_active_space( n_mo,
                                                                  frozen_indices,
                                                                  active_indices,
                                                                  virtual_indices)
    # In case we want to do a meanfield Orbital Optimisation, a better guess is
    # the so-called SAD guess : Superposition of Atomic Densities
    if SAD_guess:
        MO_guess_energy, C_guess = np.linalg.eigh( h )
        h, g = transform_1_2_body_tensors_in_new_basis( h, g, C_guess )
        # At this step the integrals are redefined with the new SAD guess

    f_min_OO = minimize( energy_cost_function_orbital_optimization,
                              x0      = Vec_k,
                              args    = ( one_rdm,
                                          two_rdm,
                                          h,
                                          g,
                                          E_rep_nuc,
                                          frozen_indices,
                                          active_indices,
                                          virtual_indices),
                              method  = method_name,
                              options = {'maxiter': max_iteration,
                                         'gtol'     : grad_tolerance,
                                         'disp': show_me}  )
    Vec_k = f_min_OO['x']
    K_mat = transform_vec_to_skewmatrix_with_active_space(Vec_k,
                                                          n_mo,
                                                          frozen_indices,
                                                          active_indices,
                                                          virtual_indices)
    U_OO =  scipy.linalg.expm( - K_mat )
    C_OO = C_ref @ U_OO
    h_OO, g_OO = transform_1_2_body_tensors_in_new_basis( h, g, U_OO )
    E_new = compute_energy_with_rdm( one_rdm,
                                    two_rdm,
                                    active_indices,
                                    h_OO,
                                    g_OO,
                                    E_rep_nuc )
    return C_OO, E_new, h_OO, g_OO



def filter_h_g_orb(Hess_OrbOrb,
                   Grad_Orb,
                   frozen_indices,
                   active_indices,
                   virtual_indices,
                   n_mo_optimized):
    """


    Parameters
    ----------
    Hess_OrbOrb     : array 
        Hessian for the orbital-orbital block
    Grad_Orb        : array
        Gradient for the orbitals
    frozen_indices  : array
        List of frozen indices
    active_indices  :  array
        List of active indices
    virtual_indices : array
        List of virtual indices
    n_mo_optimized  : int 
        number of molecular orbital to be optimized

    Returns
    -------
    Hess_OrbOrb_filtered : array
        Resulting "filtered" Hessian for the orbital-orbital block
    Grad_Orb_filtered    : array
        Resulting "filtered" Gradient for the orbital

    """
    # Counting the number of non-redundant rotation parameters
    Num_nonredundant_par = 0
    for q in range(n_mo_optimized-1):
        for p in range(q+1, n_mo_optimized):
            if not((p in frozen_indices and q in frozen_indices) or
                    (p in virtual_indices and q in virtual_indices) or
                    (p in active_indices and q in active_indices)):
                Num_nonredundant_par += 1

    # print('non redundant ', Num_nonredundant_par)
    Grad_Orb_filtered = np.zeros((Num_nonredundant_par, 1))
    Hess_OrbOrb_filtered = np.zeros(
        (Num_nonredundant_par, Num_nonredundant_par))

    ind_pq_filtered = 0
    for q in range(n_mo_optimized-1):
        for p in range(q+1, n_mo_optimized):

            if not((p in frozen_indices and q in frozen_indices) or
                    (p in virtual_indices and q in virtual_indices) or
                    (p in active_indices and q in active_indices)):

                ind_pq = get_super_index(p, q, n_mo_optimized)
                # Orbital Gradient
                Grad_Orb_filtered[ind_pq_filtered] = Grad_Orb[ind_pq]

                ind_rs_filtered = 0
                for s in range(n_mo_optimized-1):
                    for r in range(s+1, n_mo_optimized):

                        if not((r in frozen_indices and s in frozen_indices) or
                                (r in virtual_indices and s in virtual_indices) or
                                (r in active_indices and s in active_indices)):

                            ind_rs = get_super_index(r, s, n_mo_optimized)

                            # Orbital Hessian
                            Hess_OrbOrb_filtered[ind_pq_filtered,
                                                 ind_rs_filtered] = Hess_OrbOrb[ind_pq, ind_rs]
                            ind_rs_filtered += 1

                ind_pq_filtered += 1

    return Hess_OrbOrb_filtered, Grad_Orb_filtered


@njit(parallel=True)
def sa_build_mo_hessian_and_gradient(n_mo_OPTIMIZED,
                                     active_indices,
                                     frozen_indices,
                                     virtual_indices,
                                     h_MO,
                                     g_MO,
                                     F_SA,
                                     one_rdm_SA,
                                     two_rdm_SA):
    """
    Create the molecular orbital gradient and hessian necessary for orbital
    optimization with Newton-Raphson methods

    Parameters
    ----------
    n_mo_OPTIMIZED  : int 
        number of molecular orbital to be optimized
    frozen_indices  : array
        List of frozen indices
    active_indices  : array
        List of active indices
    virtual_indices : array
        List of virtual indices
    h_MO            : array
        1-electron integrals
    g_MO            : array
        2-electron integrals
    F_SA            : array
        State-averaged generalized Fock matrix
    one_rdm_SA      : array
        State-averaged 1-electron reduced density matrix
    two_rdm_SA      : array
        State-averaged 2-electron reduced density matrix

    Returns
    -------
    gradient_SA : array
        state-averaged molecular orbital gradient
    hessian_SA  : array
        state-averaged molecular orbital Hessian

    """
    gradient_SA = np.zeros((n_mo_OPTIMIZED * (n_mo_OPTIMIZED - 1) // 2, 1))
    hessian_SA = np.zeros((n_mo_OPTIMIZED * (n_mo_OPTIMIZED - 1) //
                           2, n_mo_OPTIMIZED * (n_mo_OPTIMIZED - 1) // 2))
    join_indices = frozen_indices + active_indices

    for q in prange(n_mo_OPTIMIZED - 1):
        for p in range(q + 1, n_mo_OPTIMIZED):
            ind_pq = get_super_index(p, q, n_mo_OPTIMIZED)

            # Computing the gradient vector elements
            gradient_SA[ind_pq] = 2. * (F_SA[p, q] - F_SA[q, p])

            # Continue the loop to compute the hessian matrix elements
            for s in range(n_mo_OPTIMIZED - 1):
                for r in range(s + 1, n_mo_OPTIMIZED):
                    ind_rs = get_super_index(r, s, n_mo_OPTIMIZED)

                    hessian_SA[ind_pq,
                               ind_rs] = (((F_SA[p,s]
                                          + F_SA[s,p]) * delta(q,r)
                                          - 2. * h_MO[p,s] * one_rdm_SA[q,r])
                                          - ((F_SA[q,s]
                                          + F_SA[s,q]) * delta(p,r)
                                          - 2. * h_MO[q,s] * one_rdm_SA[p,r])
                                          - ((F_SA[p,r]
                                          + F_SA[r,p]) * delta(q,s)
                                          - 2. * h_MO[p,r] * one_rdm_SA[q,s])
                                          + ((F_SA[q,r]
                                          + F_SA[r,q]) * delta(p,s)
                                          - 2. * h_MO[q,r] * one_rdm_SA[p,s]))

                    for u in join_indices:
                        for v in join_indices:
                            hessian_SA[ind_pq,ind_rs] += (
                               (2. * g_MO[p,u,r,v] * (two_rdm_SA[q,u,s,v] + two_rdm_SA[q,u,v,s])  + 2. * g_MO[p,r,u,v] * two_rdm_SA[q,s,u,v])
                             - (2. * g_MO[q,u,r,v] * (two_rdm_SA[p,u,s,v] + two_rdm_SA[p,u,v,s])  + 2. * g_MO[q,r,u,v] * two_rdm_SA[p,s,u,v])
                             - (2. * g_MO[p,u,s,v] * (two_rdm_SA[q,u,r,v] + two_rdm_SA[q,u,v,r])  + 2. * g_MO[p,s,u,v] * two_rdm_SA[q,r,u,v])
                             + (2. * g_MO[q,u,s,v] * (two_rdm_SA[p,u,r,v] + two_rdm_SA[p,u,v,r])  + 2. * g_MO[q,s,u,v] * two_rdm_SA[p,r,u,v])
                             )

    return gradient_SA, hessian_SA



@njit(parallel=True)
def sa_build_mo_hessian_and_gradient_no_active_space(h_MO,
                                                     g_MO,
                                                     F_SA,
                                                     one_rdm_SA,
                                                     two_rdm_SA):
    """
    Similar function as before but for the specific case of full active space !


    WORK IN PROGRESS ==================

    Parameters
    ----------
    h_MO            : array
        1-electron integrals
    g_MO            : array
        2-electron integrals
    F_SA            :  array
        State-averaged generalized Fock matrix
    one_rdm_SA      : array
        State-averaged 1-electron reduced density matrix
    two_rdm_SA      : array
        State-averaged 2-electron reduced density matrix

    Returns
    -------
    gradient_SA : array
        state-averaged molecular orbital gradient
    hessian_SA  : array
        state-averaged molecular orbital Hessian

    """
    n_mo_OPTIMIZED = np.shape(h_MO)[0]
    gradient_SA = np.zeros((n_mo_OPTIMIZED * (n_mo_OPTIMIZED - 1) // 2, 1))
    hessian_SA = np.zeros((n_mo_OPTIMIZED * (n_mo_OPTIMIZED - 1) //
                           2, n_mo_OPTIMIZED * (n_mo_OPTIMIZED - 1) // 2))

    for q in prange(n_mo_OPTIMIZED - 1):
        for p in range(q + 1, n_mo_OPTIMIZED):
            ind_pq = get_super_index(p, q, n_mo_OPTIMIZED)

            # Computing the gradient vector elements
            gradient_SA[ind_pq] = 2. * (F_SA[p, q] - F_SA[q, p])

            # Continue the loop to compute the hessian matrix elements
            for s in range(n_mo_OPTIMIZED - 1):
                for r in range(s + 1, n_mo_OPTIMIZED):
                    ind_rs = get_super_index(r, s, n_mo_OPTIMIZED)

                    hessian_SA[ind_pq,
                               ind_rs] = (((F_SA[p,s]
                                          + F_SA[s,p]) * delta(q,r)
                                          - 2. * h_MO[p,s] * one_rdm_SA[q,r])
                                          - ((F_SA[q,s]
                                          + F_SA[s,q]) * delta(p,r)
                                          - 2. * h_MO[q,s] * one_rdm_SA[p,r])
                                          - ((F_SA[p,r]
                                          + F_SA[r,p]) * delta(q,s)
                                          - 2. * h_MO[p,r] * one_rdm_SA[q,s])
                                          + ((F_SA[q,r]
                                          + F_SA[r,q]) * delta(p,s)
                                          - 2. * h_MO[q,r] * one_rdm_SA[p,s]))

                    for u in range(n_mo_OPTIMIZED):
                        for v in range(n_mo_OPTIMIZED):
                            hessian_SA[ind_pq,ind_rs] += (
                               (2. * g_MO[p,u,r,v] * (two_rdm_SA[q,u,s,v] + two_rdm_SA[q,u,v,s])  + 2. * g_MO[p,r,u,v] * two_rdm_SA[q,s,u,v])
                             - (2. * g_MO[q,u,r,v] * (two_rdm_SA[p,u,s,v] + two_rdm_SA[p,u,v,s])  + 2. * g_MO[q,r,u,v] * two_rdm_SA[p,s,u,v])
                             - (2. * g_MO[p,u,s,v] * (two_rdm_SA[q,u,r,v] + two_rdm_SA[q,u,v,r])  + 2. * g_MO[p,s,u,v] * two_rdm_SA[q,r,u,v])
                             + (2. * g_MO[q,u,s,v] * (two_rdm_SA[p,u,r,v] + two_rdm_SA[p,u,v,r])  + 2. * g_MO[q,s,u,v] * two_rdm_SA[p,r,u,v])
                             )

    return gradient_SA, hessian_SA


@njit(parallel=True)
def build_mo_gradient(n_mo_OPTIMIZED,
                      active_indices,
                      frozen_indices,
                      virtual_indices,
                      h_MO,
                      g_MO,
                      one_rdm,
                      two_rdm):
    """
    Create a molecular orbital gradient for an active space problem.

    Parameters
    ----------
    n_mo_OPTIMIZED  :  int 
        number of molecular orbital to be optimized
    frozen_indices  :  array 
        List of frozen indices
    active_indices  :  array 
        List of active indices
    virtual_indices :  array
        List of virtual indices
    h_MO            :  array
        1-electron integrals
    g_MO            :  array
        2-electron integrals
    one_rdm         :  array
        1-electron reduced density matrix
    two_rdm         :  array
        2-electron reduced density matrix

    Returns
    -------
    gradient        :  array
        molecular orbital gradient

    """
    Num_MO = np.shape(h_MO)[0]
    gradient = np.zeros((n_mo_OPTIMIZED * (n_mo_OPTIMIZED - 1) // 2, 1))
    F = build_generalized_fock_matrix(Num_MO,
                                        h_MO,
                                        g_MO,
                                        one_rdm,
                                        two_rdm,
                                        active_indices,
                                        frozen_indices)

    # join = active_indices + frozen_indices
    for q in prange(n_mo_OPTIMIZED - 1):
        for p in range(q + 1, n_mo_OPTIMIZED):
            ind_pq = get_super_index(p, q, n_mo_OPTIMIZED)
            gradient[ind_pq] = 2. * (F[p, q] - F[q, p])

    return gradient


def orbital_optimisation_newtonraphson(one_rdm_SA,
                                       two_rdm_SA,
                                       active_indices,
                                       frozen_indices,
                                       virtual_indices,
                                       C_transf,
                                       E_rep_nuc,
                                       h_AO,
                                       g_AO,
                                       n_mo_optimized,
                                       OPT_OO_MAX_ITER=100,
                                       Grad_threshold=1e-6,
                                       TELL_ME=True):
    """
    Orbital optimization with a Newton-Raphson method for an active space problem. 

    Parameters
    ----------
    one_rdm_SA      : array 
        State-averaged 1-electron reduced density matrix
    two_rdm_SA      : array
        State-averaged 2-electron reduced density matrix
    frozen_indices  :  array
        List of frozen indices
    active_indices  :  array
        List of active indices
    virtual_indices :  array
        List of virtual indices
    C_transf        : array
        Initial MO coeff matrix
    E_rep_nuc       :  array
        Energy of the nuclei repulsion
    h_AO            :  array
        1-electron integrals in the AO basis
    g_AO            :  array
        2-electron integrals in the AO basis
    n_mo_optimized  :  int 
        Number of molecular orbitaled to be optimized
    OPT_OO_MAX_ITER : int
        Maximum number of Newton-Raphson iterations (i.e. steps). The default is 100.
    Grad_threshold  :  float 
        Thershold of the gradient to define the optimization convergence. The default is 1e-6.
    TELL_ME         :  boolean 
        To show or not the evolution of the optimization process.
                       The default is True.

    Returns
    -------
    C_transf_best_OO :  array
        Orbital optimized orbital coefficient matrix
    E_best_OO        :  float 
        Orbital optimized energy
    h_best           :  array
        Orbital optimized 1-electron orbital integral
    g_best           :  array
        Orbital optimized 2-electro integral

    """

    h_MO, g_MO = transform_1_2_body_tensors_in_new_basis( h_AO, g_AO, C_transf )
    n_mo = np.shape(h_MO)[0]
    E_old_OO = compute_energy_with_rdm(one_rdm_SA,
                                        two_rdm_SA,
                                        active_indices,
                                        h_MO,
                                        g_MO,
                                        E_rep_nuc)
    E_new_OO = 1e+99
    E_best_OO = E_old_OO
    C_transf_best_OO = C_transf
    h_best = h_MO
    g_best = g_MO
    k_vec = np.zeros((n_mo_optimized * (n_mo_optimized - 1) // 2, 1))
    C_ref = C_transf

    # Printing the evolution of the Orb Opt process
    if TELL_ME:
        print("ENERGY    |    ORBITAL GRADIENT  | RATIO PREDICTION  | ITERATION")
        print(E_best_OO)
    for iteration in range(OPT_OO_MAX_ITER):

        # Building the state-averaged generalized Fock matrix
        # F_SA = build_generalized_fock_matrix_active_space_adapted(n_mo,
        #                                     h_MO,
        #                                     g_MO,
        #                                     one_rdm_SA,
        #                                     two_rdm_SA,
        #                                     active_indices,
        #                                     frozen_indices)

        F_SA = build_generalized_fock_matrix(n_mo,
                                             h_MO,
                                             g_MO,
                                             one_rdm_SA,
                                             two_rdm_SA )

        # Building the State-Averaged Gradient and Hessian for the two states
        SA_Grad, SA_Hess = sa_build_mo_hessian_and_gradient(n_mo_optimized,
                                                            active_indices,
                                                            frozen_indices,
                                                            virtual_indices,
                                                            h_MO,
                                                            g_MO,
                                                            F_SA,
                                                            one_rdm_SA,
                                                            two_rdm_SA)

        Grad_norm = np.linalg.norm( SA_Grad )

        SA_Hess_filtered, SA_Grad_filtered = filter_h_g_orb(SA_Hess,
                                                            SA_Grad,
                                                            frozen_indices,
                                                            active_indices,
                                                            virtual_indices,
                                                            n_mo_optimized)

        Grad_norm = np.linalg.norm(SA_Grad_filtered)

        # AUGMENTED HESSIAN APPROACH =====
        Aug_Hess = np.block([[0., SA_Grad_filtered.T],
                             [SA_Grad_filtered, SA_Hess_filtered]])

        Eig_val_Aug_Hess, Eig_vec_Aug_Hess = np.linalg.eigh(Aug_Hess)
        step_k = np.reshape( Eig_vec_Aug_Hess[1:, 0] / Eig_vec_Aug_Hess[0, 0], np.shape(SA_Grad_filtered) )

        if (np.max(np.abs(step_k)) > 0.05):
            step_k = 0.05 * step_k / np.max(np.abs(step_k))

        step_k_reshaped = np.zeros(( n_mo_optimized * (n_mo_optimized - 1) // 2, 1 ))
        ind_pq_filtered = 0
        for q in range(n_mo_optimized - 1):
            for p in range(q + 1, n_mo_optimized):
                ind_pq = get_super_index(p, q, n_mo_optimized)
                if not((p in frozen_indices  and q in frozen_indices) or
                       (p in virtual_indices and q in virtual_indices) or
                       (p in active_indices  and q in active_indices)):
                    step_k_reshaped[ind_pq] = step_k[ind_pq_filtered]
                    ind_pq_filtered += 1

        step_k = step_k_reshaped

        # Building the Rotation operator with a Netwon-Raphson Step
        # Updating the rotation vector "k"
        k_vec = k_vec + step_k

        # Generator of the rotation : the kew-matrix K = skew(k)
        K_mat = transform_vec_to_skewmatrix(k_vec, n_mo_optimized)
        # print( K_mat )

        # Rotation operator in the MO basis : U = exp(-K)
        U_OO = ( scipy.linalg.expm( - K_mat ) ).real

        # Completing the transformation operator : in case not all the MOs are considered
        # in the OO process, we extend the operator with an identity block
        if (n_mo_optimized < n_mo):
            U_OO = scipy.linalg.block_diag(
                U_OO, np.identity(n_mo - n_mo_optimized))

        # Transforming the MO coeff matrix to encode the optimized MOs
        C_transf = C_ref @ U_OO

        # Building the resulting modified MO integrals for the next cycle of  calculation
        h_MO, g_MO = transform_1_2_body_tensors_in_new_basis( h_AO, g_AO, C_transf )

        # Computing the resulting orbital-Optimized energy
        E_new_OO = compute_energy_with_rdm(one_rdm_SA,
                                            two_rdm_SA,
                                            active_indices,
                                            h_MO,
                                            g_MO,
                                            E_rep_nuc)
        # print("wTF ", E_new_OO)
        # Checking the accuracy of the energy prediction
        Ratio = (E_new_OO - E_old_OO) / (SA_Grad.T @
                                         step_k + 0.5 * step_k.T @ SA_Hess @ step_k)

        # Storing the best data obtained during the optimization
        if ( E_new_OO < E_best_OO ):
            E_best_OO = E_new_OO
            C_transf_best_OO = C_transf
            h_best = h_MO
            g_best = g_MO
            if TELL_ME:
                print(E_new_OO, Grad_norm, Ratio, iteration, " +++ ")
        else:
            if TELL_ME:
                print(E_new_OO, Grad_norm, Ratio, iteration)

        E_old_OO = E_new_OO

        if Grad_norm < Grad_threshold:
            break

    return C_transf_best_OO, E_best_OO, h_best, g_best



def orbital_optimisation_newtonraphson_no_active_space(one_rdm_SA,
                                                       two_rdm_SA,
                                                       C_transf,
                                                       E_rep_nuc,
                                                       h_AO,
                                                       g_AO,
                                                       n_mo_optimized,
                                                       OPT_OO_MAX_ITER,
                                                       Grad_threshold,
                                                       TELL_ME=True):
    """
    Similar function as before but for the specific case of full active space !

    Parameters
    ----------
    one_rdm_SA      : array
        State-averaged 1-electron reduced density matrix
    two_rdm_SA      :  array
        State-averaged 2-electron reduced density matrix
    C_transf        : array
        Initial MO coeff matrix
    E_rep_nuc       : float
        Energy of the nucleic repulsion
    h_AO            : array
        1-electron integrals in the AO basis
    g_AO            : array
        2-electron integrals in the AO basis
    n_mo_optimized  : int
        Number of molecular orbitaled to be optimized
    OPT_OO_MAX_ITER : int
        Maximum number of Newton-Raphson iterations (i.e. steps). The default is 100.
    Grad_threshold  : float 
        Thershold of the gradient to define the optimization convergence. The default is 1e-6.
    TELL_ME         : boolean
        To show or not the evolution of the optimization process. The default is True.

    Returns
    -------
    C_transf_best_OO : array
        Orbital optimized orbital coefficient matrix
    E_best_OO        : array
        Orbital optimized energy
    h_best           :  array
        Orbital optimized 1-electron orbital integral
    g_best           : array
        Orbital optimized 2-electro integral

    """

    h_MO, g_MO = transform_1_2_body_tensors_in_new_basis( h_AO, g_AO, C_transf )

    SAD_guess = False
    if SAD_guess:
        MO_guess_energy, C_guess = np.linalg.eigh( h_MO )
        h_MO, g_MO = transform_1_2_body_tensors_in_new_basis( h_MO, g_MO, C_guess )

    n_mo = np.shape(h_MO)[0]
    active_indices = [i for i in range(n_mo)]
    E_old_OO = compute_energy_with_rdm(one_rdm_SA,
                                        two_rdm_SA,
                                        active_indices,
                                        h_MO,
                                        g_MO,
                                        E_rep_nuc)
    E_new_OO = 1e+99
    E_best_OO = E_old_OO
    C_transf_best_OO = C_transf
    h_best = h_MO
    g_best = g_MO
    k_vec = np.zeros((n_mo_optimized * (n_mo_optimized - 1) // 2, 1))
    C_ref = C_transf
    step_k = 0.

    # Printing the evolution of the Orb Opt process
    if TELL_ME:
        print("ENERGY    |    ORBITAL GRADIENT  | RATIO PREDICTION  | ITERATION")
        print(E_best_OO)

    for iteration in range(OPT_OO_MAX_ITER):

        # Building the state-averaged generalized Fock matrix
        F_SA = build_generalized_fock_matrix(n_mo,
                                             h_MO,
                                             g_MO,
                                             one_rdm_SA,
                                             two_rdm_SA )

        # Building the State-Averaged Gradient and Hessian for the two states
        SA_Grad, SA_Hess = sa_build_mo_hessian_and_gradient_no_active_space(h_MO,
                                                                            g_MO,
                                                                            F_SA,
                                                                            one_rdm_SA,
                                                                            two_rdm_SA)

        Grad_norm = np.linalg.norm( SA_Grad )

        SA_Grad_filtered = SA_Grad
        SA_Hess_filtered = SA_Hess

        # AUGMENTED HESSIAN APPROACH =====
        Aug_Hess = np.block([[0., SA_Grad_filtered.T],
                             [SA_Grad_filtered, SA_Hess_filtered]])

        Eig_val_Aug_Hess, Eig_vec_Aug_Hess = np.linalg.eigh(Aug_Hess)
        # step_k = np.reshape( Eig_vec_Aug_Hess[1:, 0] / Eig_vec_Aug_Hess[0, 0], np.shape(SA_Grad_filtered) )

        step_k =  - np.linalg.inv( SA_Hess_filtered ) @ SA_Grad_filtered
        # if (np.max(np.abs(step_k)) > 0.05):
        #     step_k = 0.05 * step_k / np.max(np.abs(step_k))

        # Building the Rotation operator with a Netwon-Raphson Step
        # Updating the rotation vector "k"
        k_vec = k_vec + step_k

        # Generator of the rotation : the kew-matrix K = skew(k)
        K_mat = transform_vec_to_skewmatrix(k_vec, n_mo_optimized)

        # Rotation operator in the MO basis : U = exp(-K)
        U_OO = ( scipy.linalg.expm( - K_mat ) ).real

        # Completing the transformation operator : in case not all the MOs are considered
        # in the OO process, we extend the operator with an identity block
        if (n_mo_optimized < n_mo):
            U_OO = scipy.linalg.block_diag(
                U_OO, np.identity(n_mo - n_mo_optimized))

        # Transforming the MO coeff matrix to encode the optimized MOs
        C_transf = C_ref @ U_OO

        # Building the resulting modified MO integrals for the next cycle of  calculation
        h_MO, g_MO = transform_1_2_body_tensors_in_new_basis( h_AO, g_AO, C_transf )

        # Computing the resulting orbital-Optimized energy
        E_new_OO = compute_energy_with_rdm(one_rdm_SA,
                                            two_rdm_SA,
                                            active_indices,
                                            h_MO,
                                            g_MO,
                                            E_rep_nuc)
        # print("wTF ", E_new_OO)
        # Checking the accuracy of the energy prediction
        Ratio = (E_new_OO - E_old_OO) / (SA_Grad.T @
                                         step_k + 0.5 * step_k.T @ SA_Hess @ step_k)

        # Storing the best data obtained during the optimization
        if ( E_new_OO < E_best_OO ):
            E_best_OO = E_new_OO
            C_transf_best_OO = C_transf
            h_best = h_MO
            g_best = g_MO
            if TELL_ME:
                print(E_new_OO, Grad_norm, Ratio, iteration, " +++ ")
        else:
            if TELL_ME:
                print(E_new_OO, Grad_norm, Ratio, iteration)

        E_old_OO = E_new_OO

        if Grad_norm < Grad_threshold:
            break

    return C_transf_best_OO, E_best_OO, h_best, g_best


def transform_vec_to_skewmatrix(Vec_k, n_mo):
    """
    Create the anti-symmetric K matrix necessary for the orbital optimization
    based on a vector k encoding the kappa (i.e. orbital rotation parameters)

    Parameters
    ----------
    Vec_k :  array
        vector encoding the orbital rotation parameters
    n_mo  :  int
        number of orbital

    Returns
    -------
    Skew_Matrix_K : array
        Anti-symmetric K matrix for the orbital optimization

    """
    Skew_Matrix_K = np.zeros((n_mo, n_mo))
    ind_ij = 0
    for j in range(n_mo-1):
        for i in range(j+1, n_mo):
            Skew_Matrix_K[i, j] = Vec_k[ind_ij]
            Skew_Matrix_K[j, i] = - Vec_k[ind_ij]
            ind_ij += 1
    return Skew_Matrix_K


@njit
def build_generalized_fock_matrix(Num_MO,
                                  h_MO,
                                  g_MO,
                                  one_rdm,
                                  two_rdm ):
    """
    Create the generalized fock matrix with no assumption of active space 
    (i.e. full treatment with no simplifcation in the building)

    Parameters
    ----------
    Num_MO  : int
        Number of molecular orbital
    h_MO    : array
        1-electron integrals
    g_MO    : array
        2-electron integrals
    one_rdm : array
        1-electron reduced density matrix
    two_rdm : array
        2-electron reduced density matrix

    Returns
    -------
    F :  array 
        Generalized Fock matrix

    """
    F = np.zeros((Num_MO, Num_MO))
    for m in range(Num_MO):
        for n in range(Num_MO):
            for q in range(Num_MO):
                 F[m,n] += one_rdm[m, q] * h_MO[n,q]
                 for r in range(Num_MO):
                     for s in range(Num_MO):
                         F[m,n] += two_rdm[m,q,r,s] * g_MO[n,q,r,s]
    return F


@njit
def build_generalized_fock_matrix_active_space_adapted(Num_MO,
                                                      h_MO,
                                                      g_MO,
                                                      one_rdm,
                                                      two_rdm,
                                                      active_indices,
                                                      frozen_indices):
    """
    Create a generalized fock matrix for a system with an active space.
    It makes it possible to use lots of simplification in this specific case.

    Parameters
    ----------
    Num_MO          : array
        Number of molecular orbital
    h_MO            : array
        1-electron integrals
    g_MO            : array
        2-electron integrals
    one_rdm         : array
        1-electron redcued density matrix
    two_rdm         : array
        2-electron redcued density matrix
    active_indices  : array
        List of active space indices
    frozen_indices  : array
        List of frozen space indices

    Returns
    -------
    F : array 
        Generalized fock matrix

    """
    F_I = f_inactive(Num_MO, h_MO, g_MO, frozen_indices)
    F_A = f_active(Num_MO, g_MO, one_rdm, active_indices)
    Q = q_aux(Num_MO, g_MO, two_rdm, active_indices)

    F = np.zeros((Num_MO, Num_MO))
    for m in range(Num_MO):
        for n in range(Num_MO):

            if (frozen_indices is not None):
                if m in frozen_indices:
                    F[m, n] = 2. * (F_I[n, m] + F_A[n, m])

            elif m in active_indices:
                F[m, n] = Q[m, n]
                for w in active_indices:
                    F[m, n] += F_I[n, w] * one_rdm[m, w]
    return F


@njit
def f_inactive(Num_MO, h_MO, g_MO, frozen_indices):
    """
    Create the inactive part Fock matrix contribution

    Parameters
    ----------
    Num_MO          : Number of molecular orbitals
    h_MO            : 1-electron integral
    g_MO            : 2-electron integral
    frozen_indices  : List of frozen indices

    Returns
    -------
    F_inactive      : Inactive Fock matrix

    """

    F_inactive = np.zeros((Num_MO, Num_MO))
    for m in range(Num_MO):
        for n in range(Num_MO):
            F_inactive[m, n] = h_MO[m, n]
            if (frozen_indices is not None):
                for i in frozen_indices:
                    F_inactive[m, n] += 2. * g_MO[m, n, i, i] - g_MO[m, i, i, n]
    return F_inactive


@njit
def f_active(Num_MO, g_MO, one_rdm, active_indices):
    """
    Create the active part Fock matrix contribution

    Parameters
    ----------
    Num_MO          : number of molecular orbitals
    g_MO            : 2-electron reduced density matrix
    one_rdm         : 1-electorn reduced density matrix
    active_indices  : list of active indices

    Returns
    -------
    F_active :  Active Fock matrix

    """
    F_active = np.zeros((Num_MO, Num_MO))
    for m in range(Num_MO):
        for n in range(Num_MO):
            for v in active_indices:
                for w in active_indices:
                    F_active[m, n] += one_rdm[v, w] * \
                        (g_MO[m, n, v, w] - 0.5 * g_MO[m, w, v, n])
    return F_active


@njit
def q_aux(Num_MO, g_MO, two_rdm, active_indices):
    """
    Auxiliarry matrix for the generalized Fock matrix resolution
    (From Helgaker's PINK BOOK)

    Parameters
    ----------
    Num_MO          : Number of molecualr orbitals
    g_MO            : 2-electron integrals
    two_rdm         : 2-electorn reduced density matrix
    active_indices  : List of active indices

    Returns
    -------
    Q_aux           : Auxiliarry matrix

    """
    Q_aux = np.zeros((Num_MO, Num_MO))
    for v in active_indices:
        for m in range(Num_MO):
            for w in active_indices:
                for x in active_indices:
                    for y in active_indices:
                        Q_aux[v, m] += two_rdm[v, w, x, y] * g_MO[m, w, x, y]
    return Q_aux


@njit
def get_super_index(p, q, n_mo):
    """
    Create super index baed on a composition of two indices. Improtant for
    orbital optimization methods

    Parameters
    ----------
    p    :  Molecular orbital index "p"
    q    :  Molecular orbital index "q"
    n_mo :  Number of molecular orbital

    Returns
    -------
    ind_pq :  Super index

    """
    ini_int = n_mo-1 - q
    fin_int = n_mo-1
    counter = (fin_int-ini_int+1)*(ini_int+fin_int)//2
    ind_pq = counter + p - n_mo
    return ind_pq


@njit
def delta(index_1, index_2):
    """
    Create a kronecker delta based on two indices

    Parameters
    ----------
    index_1 : First index
    index_2 : Second index

    Returns
    -------
    d :  Result of the Konecker's delta

    """
    d = 0.0
    if index_1 == index_2:
        d = 1.0
    return d


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


# =============================================================================
# HOUSEHOLDER TRANSFORMATIONS
# =============================================================================

def householder_transformation(M):
    """
    Householder transformation transforming a squared matrix " M " into a
    block-diagonal matrix " M_BD " such that

    .. math::  
        M_BD = P M P

    where " P " represents the Householder transformation built from the
    vector "v" such that

    .. math:: 
        P = I - 2  v . v^T

    .. note:: 
        This returns a 2x2 block on left top corner

    Parameters
    ----------
    M : array
        Squared matrix to be transformed

    Returns
    -------
    P :  array
        Transformation matrix
    v :  array
        Householder vector

    """
    n = np.shape(M)[0]
    if np.count_nonzero(M - np.diag(np.diagonal(M))) == 0:
        print('\tinput was diagonal matrix')
        return np.diag(np.zeros(n) + 1), np.zeros((n, 1))
    # Build the Householder vector "H_vector"
    alpha = - np.sign(M[1, 0]) * sum(M[j, 0] ** 2. for j in range(1, n)) ** 0.5
    r = (0.5 * (alpha ** 2. - alpha * M[1, 0])) ** 0.5

    vector = np.zeros((n, 1))
    vector[1] = (M[1, 0] - alpha) / (2.0 * r)
    for j in range(2, n):
        vector[j] = M[j, 0] / (2.0 * r)

    # Building the transformation matrix "P"
    P = np.eye(n) - 2 * vector @ vector.T

    return P, vector


def block_householder_transformation(M, size):
    """
    Block Householder transformation transforming a square matrix ” M ” into a
    block-diagonal matrix ” M_BD ” such that

    .. math:: 
        M_{BD} = H(V) M H(V)

    where ” H(V) ” represents the Householder transformation built from the
    matrix “V” such that

    .. math:: 
        H(V) = I - 2  V(V^{T} V)^{-1}V^{T}

    .. note:: 
        Depending on the size of the block needed (unchanged by the transformation),
        this returns a (2 x size)*(2 x size) block on left top corner

    See Also
    --------
    Article: F. Rotella, I. Zambettakis, Block Householder Transformation for Parallel QR Factorization,
      Applied Mathematics Letter 12 (1999) 29-34

    Parameters
    ----------
    M :  array
        Square matrix to be transformed
    size : array
        size of the block ( must be > 1)

    Returns
    -------
    P   : array
        Transformation matrix
    moore_penrose_inv : array
        Moore Penrose inverse of Householder matrix

    """
    n = np.shape(M)[0]
    a1 = M[size:2 * size, :size]
    a1_inv = np.linalg.inv(a1)
    a2 = M[2 * size:, :size]
    a2_a1_inv = a2 @ a1_inv
    a2_a1_inv_tr = np.transpose(a2_a1_inv)
    a3 = np.eye(size) + a2_a1_inv_tr @ a2_a1_inv
    eigval, eigvec = np.linalg.eig(a3)
    # eigvec v[:,i] corresponds to eigval w[i]
    eigval = np.diag(eigval)
    eigvec_tr = np.transpose(eigvec)
    x_d = eigvec @ (eigval ** 0.5) @ eigvec_tr
    x = x_d @ a1
    y = a1 + x
    v = np.block([[np.zeros((size, size))], [y], [M[2 * size:, :size]]])
    v_tr = np.transpose(v)
    v_tr_v = v_tr @ v
    v_tr_v_inv = np.linalg.inv(v_tr_v)
    moore_penrose_inv = v_tr_v_inv @ v_tr
    P = np.eye(n) - 2. * v @ v_tr_v_inv @ v_tr
    return P, moore_penrose_inv


# =============================================================================
#  A FEW PREMADE GEOMETRIES
# =============================================================================

def generate_h_chain_geometry(N_atoms, dist_HH):
    """
    A function to build a Hydrogen chain geometry (on the x-axis)

    Parameters
    ----------
    N_atoms  : int
        Total number of Hydrogen atoms
    dist_HH   : float
        Distance between two consecutive atom of the ring

    Returns
    ----------
    h_chain_geometry : str
        XYZ string encoding the geometry of the ring chain

    """
    h_chain_geometry = 'H 0. 0. 0.'  # Define the position of the first atom
    for n in range(1, N_atoms):
        h_chain_geometry += '\nH {} 0. 0.'.format(n * dist_HH)

    return h_chain_geometry


def generate_h_ring_geometry(N_atoms, radius):
    """
    A function to build a Hydrogen ring geometry (in the x-y plane)

    Parameters
    ----------
    N_atoms  : int
        Total number of Hydrogen atoms
    radius   : float
        Radius of the ring

    Returns
    ----------
    h_ring_geometry : str
        XYZ string encoding the geometry of the hydrogen ring

    """
    theta_hh = 2 * np.pi / N_atoms  # Angle separating two consecutive H atoms (homogeneous distribution)
    theta_ini = 0.0
    h_ring_geometry = '\nH {:.16f} {:.16f} 0.'.format(radius * np.cos(theta_ini), radius * np.sin(
        theta_ini))  # 'H {:.16f} 0. 0.'.format( radius )  # Define the position of the first atom
    for n in range(1, N_atoms):
        angle_h = theta_ini + theta_hh * n
        h_ring_geometry += '\nH {:.16f} {:.16f} 0.'.format(radius * np.cos(angle_h), radius * np.sin(angle_h))

    return h_ring_geometry


def generate_h4_geometry( radius, angle ):
    """
    A function to build a Hydrogen rectangle geometry (in the x-y plane)

    Parameters
    ----------
    angle    : float
        angle between the two right (and left) couple of atoms
    radius   : float
        Radius of the ring

    Returns
    ----------
    h_ring_geometry : str
        XYZ string encoding the geometry of the hydrogen ring

    """
    h4_geometry = """ H   {0}   {1}  0.
                      H   {0}  -{1}  0.
                      H  -{0}   {1}  0.
                      H  -{0}  -{1}  0. """.format( radius*np.cos(angle/2.), radius*np.sin(angle/2.) )
    return h4_geometry
