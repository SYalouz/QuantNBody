import scipy
from scipy.optimize import minimize
import numpy as np
from itertools import combinations
from numba import njit, prange 
import psi4 

E_ = False
e_ = False
 
# =============================================================================
# CORE FUNCTIONS FOR THE BUILDING OF the "A_dagger A" OPERATOR
# =============================================================================

def build_nbody_basis(n_mo, N_electron, S_z_cleaning=False):
    """
    Create a many-body basis formed by a list of slater-determinants
    (i.e. their occupation number)

    Parameters
    ----------
    n_mo         :  Number of molecular orbitals
    N_electron   :  Number of electrons
    S_z_cleaning :  Option if we want to get read of the s_z != 0 states (default is False)

    Returns
    -------
    nbody_basis :  List of many-body states (occupation number states) in the basis (occupation number vectors)
    """
    # Building the N-electron many-body basis
    nbody_basis = []
    for combination in combinations(range(2 * n_mo), N_electron):
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
    Return the value fo the S_z operator for a unique slater determinant

    Parameters
    ----------
    ref_state :  Slater determinant (list of occupation numbers)

    Returns
    -------
    s_z_slater_determinant : value of S_z for the given slater determinant
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

                if MO_p == MO_q:  # <=== Necessary to build the Spins operator but not really for Hamiltonians

                    # Single excitation : spin beta -- alpha
                    p, q = 2 * MO_p + 1, 2 * MO_p
                    if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
                        pass
                    elif ref_state[q] == 1:
                        kappa_, p1, p2 = build_final_state_ad_a(np.array(ref_state), p, q, mapping_kappa)
                        a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] = p1 * p2

                        # Single excitation : spin alpha -- beta
                    p, q = 2 * MO_p, 2 * MO_p + 1

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


@njit
def build_mapping(nbody_basis):
    """
    Function to create a unique mapping between a kappa vector and an occupation
    number state.

    Parameters
    ----------
    nbody_basis :  Many-

    Returns
    -------
    mapping_kappa : List of unique values associated to each kappa
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
    Function to translate a slater determinant into an unique integer

    Parameters
    ----------
    ref_state : Reference slater determinant to turn out into an integer

    Returns
    -------
    number : unique integer referring to the slater determinant
    """
    number = 0
    for digit in range(len(ref_state)):
        number += ref_state[digit] * 2 ** (len(ref_state) - digit - 1)

    return number


@njit
def new_state_after_sq_fermi_op(type_of_op, index_mode, ref_fock_state):
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
    new_fock_state = ref_fock_state.copy()
    coeff_phase = (-1.) ** np.sum(ref_fock_state[0:index_mode])
    if type_of_op == 'a':
        new_fock_state[index_mode] += -1
    elif type_of_op == 'a^':
        new_fock_state[index_mode] += 1

    return new_fock_state, coeff_phase


@njit
def build_final_state_ad_a(ref_state, p, q, mapping_kappa):
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
                                        penalty=100):
    """
    Create a matrix representation of the electronic structure Hamiltonian in any
    extended many-body basis

    Parameters
    ----------
    h_          :  One-body integrals
    g_          :  Two-body integrals
    nbody_basis :  List of many-body states (occupation number states)
    a_dagger_a  :  Matrix representation of the a_dagger_a operator
    S_2         :  Matrix representation of the S_2 operator (default is None)
    S_2_target  :  Value of the S_2 mean value we want to target (default is None)
    penalty     :  Value of the penalty term for state not respecting the spin symmetry (default is 100).

    Returns
    -------
    H_chemistry :  Matrix representation of the electronic structure Hamiltonian

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

    for p in range(n_mo):
        for q in range(n_mo):
            H_chemistry += E_[p, q] * h_[p, q]
            for r in range(n_mo):
                for s in range(n_mo):
                    e_[p, q, r, s] = E_[p, q] @ E_[r, s]
                    if q == r:
                        e_[p, q, r, s] += - E_[p, s]

                    H_chemistry += e_[p, q, r, s] * g_[p, q, r, s] / 2.

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
                                    v_term=None):
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
    H_fermi_hubbard :  Matrix representation of the Fermi-Hubbard Hamiltonian

    """
    # # Dimension of the problem 
    dim_H = len(nbody_basis)
    n_mo = np.shape(h_)[0]

    global E_
    E_ = np.empty((n_mo, n_mo), dtype=object)

    # Building the N-electron Fermi-Hubbard matrix hamiltonian (Sparse)
    H_fermi_hubbard = scipy.sparse.csr_matrix((dim_H, dim_H))
    for p in range(n_mo):
        for q in range(n_mo):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]
            H_fermi_hubbard += E_[p, q] * h_[p, q]
            for r in range(n_mo):
                for s in range(n_mo):
                    if U_[p, q, r, s] != 0:  # if U is 0, it doesn't make sense to multiply matrices
                        H_fermi_hubbard += a_dagger_a[2 * p, 2 * q] @ a_dagger_a[2 * r + 1, 2 * s + 1] * U_[p, q, r, s]
                        
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

    return H_fermi_hubbard


def build_penalty_orbital_occupancy( a_dagger_a, occupancy_target ):
    """
    Create a penalty operator to enforce a given occupancy number of electron
    in the molecular orbitals of the system

    Parameters
    ----------
    a_dagger_a       : Matrix representation of the a_dagger_a operator
    occupancy_target : Target number of electrons in a given molecular orbital
    Returns
    -------
    occupancy_penalty        : penalty operator
    list_good_states_indices : list of many body state indices respecting the constraint
    list_bad_states_indices  : list of many body state indices not respecting the constraint
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
    Build the spin-free E_pq and e_pqrs many-body operators for quantum chemistry 

    Parameters
    ----------
    a_dagger_a       : Matrix representation of the a_dagger_a operator
    n_mo             : Number of molecular orbitals considered
    Returns
    -------
    E_, e_           : The spin-free E_pq and e_pqrs many-body operators
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
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build the MO 1/2-ELECTRON DENSITY MATRICES from a
    reference wavefunction expressed in the computational basis
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    """ 
    if active_indices is not None:
        first_act_index = active_indices[0] 
        n_active_mo = len( active_indices )
        print("THATS A TEST",first_act_index)
    
    one_rdm_a = np.zeros((n_mo_total, n_mo_total))
    two_rdm_a = np.zeros((n_mo_total, n_mo_total, n_mo_total, n_mo_total))
    
    global E_
    global e_

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
                one_rdm_a[i, j] = 2. * delta(i, j)
                for k in frozen_indices:
                    for l in frozen_indices:
                        # State A
                        two_rdm_a[i, j, k, l] = 4. * delta(i, j) * delta(k, l) - 2. * delta(i, l) * delta(j, k)
    
    if active_indices is not None:
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
    
                if frozen_indices is not None:
                    # 2-RDM elements between the active and frozen spaces
                    for i in frozen_indices:
                        for j in frozen_indices:
                            # State A
                            two_rdm_a[i, j, p, q] = two_rdm_a[p, q, i, j] = 2. * delta(i, j) * one_rdm_a[p, q]
                            two_rdm_a[p, i, j, q] = two_rdm_a[j, q, p, i] = - delta(i, j) * one_rdm_a[p, q]

    return one_rdm_a, two_rdm_a



def build_1rdm_alpha(WFT, a_dagger_a):
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


def build_1rdm_beta(WFT, a_dagger_a):
    """
    Create a spin-beta 1 RDM out of a given wave function

    Parameters
    ----------
    WFT        :  Wave function for which we want to build the 1-RDM
    a_dagger_a :  Matrix representation of the a_dagger_a operator

    Returns
    -------
    One_RDM_alpha : Spin-beta 1-RDM

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
    Create a spin-free 1 RDM out of a given wave function

    Parameters
    ----------
    WFT        :  Wave function for which we want to build the 1-RDM
    a_dagger_a :  Matrix representation of the a_dagger_a operator

    Returns
    -------
    one_rdm : Spin-free 1-RDM

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
                        two_rdm_fh[p, q, r, s] += WFT.T @ a_dagger_a[2 * p, 2 * q] @ a_dagger_a[2 * r + 1, 2 * s + 1] \
                                                  @ WFT
    return two_rdm_fh


def build_2rdm_fh_dipolar_interactions(WFT, a_dagger_a, mask=None):
    """
    Create a spin-free 2 RDM out of a given Fermi Hubbard wave function for the diplar interaction operator
    it corresponds to <psi|(a^+_i↑ a_j↑ + a^+_i↓ a_j↓)(a^+_k↑ a_l↑ + a^+_k↓ a_l↓)|psi>

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
    One_RDM_alpha : Spin-free 1-RDM

    """
    n_mo = np.shape(a_dagger_a)[0] // 2
    two_rdm_fh = np.zeros((n_mo, n_mo, n_mo, n_mo))
    big_e_ = np.empty((2 * n_mo, 2 * n_mo), dtype=object)
    for p in range(n_mo):
        for q in range(n_mo):
            big_e_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]
    for p in range(n_mo):
        for q in range(n_mo):
            for r in range(n_mo):
                for s in range(n_mo):
                    if mask is None or mask[p, q, r, s] != 0:
                        two_rdm_fh[p, q, r, s] = WFT.T @ big_e_[p, q] @ big_e_[r, s] @ WFT
    return two_rdm_fh


def build_2rdm_spin_free(WFT, a_dagger_a):
    """
    Create a spin-free 2 RDM out of a given wave function

    Parameters
    ----------
    WFT        :  Wave function for which we want to build the 1-RDM
    a_dagger_a :  Matrix representation of the a_dagger_a operator

    Returns
    -------
    One_RDM_alpha : Spin-free 1-RDM

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
    Create a spin-free 2 RDM out of a given wave function

    Parameters
    ----------
    WFT        :  Wave function for which we want to build the 1-RDM
    a_dagger_a :  Matrix representation of the a_dagger_a operator

    Returns
    -------
    One_RDM_alpha : Spin-free 1-RDM

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


def my_state(slater_determinant, nbody_basis):
    """
    Translate a Slater determinant (occupation number list) into a many-body
    state referenced into a given Many-body basis.

    Parameters
    ----------
    slater_determinant  : occupation number list
    nbody_basis   : List of many-body states (occupation number states)

    Returns
    -------
    state :  The slater determinant referenced in the many-body basis
    """
    kappa = np.flatnonzero((nbody_basis == slater_determinant).all(1))[0]  # nbody_basis.index(slater_determinant)
    state = np.zeros(np.shape(nbody_basis)[0])
    state[kappa] = 1.

    return state


# =============================================================================
#   INTEGRALS BUILDER FOR ACTIVE SPACE CALCULATION
# =============================================================================

def fh_get_active_space_integrals(h_MO, U_MO, frozen_indices=None, active_indices=None):
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
        core_energy += 2 * h_MO[i, i]
        for j in frozen_indices:
            core_energy += U_MO[i, i, j, j]

    # Modified one-electron integrals
    h_act = h_MO.copy()
    for t in active_indices:
        for u in active_indices:
            for i in frozen_indices:
                h_act[t, u] += U_MO[i, i, t, u]

    return (core_energy,
            h_act[np.ix_(active_indices, active_indices)],
            U_MO[np.ix_(active_indices, active_indices, active_indices, active_indices)])


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
#  DIFFERENT TYPES OF SPIN OPERATORS
# =============================================================================

def build_s2_sz_splus_operator(a_dagger_a):
    """
    Create a matrix representation of the spin operators s_2, s_z and s_plus
    in the many-body basis.

    Parameters
    ----------
    a_dagger_a : matrix representation of the a_dagger_a operator in the many-body basis.

    Returns
    -------
    s_2, s_plus, s_z :  matrix representation of the s_2, s_plus and s_z operators
                        in the many-body basis.
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



def build_s2_local( a_dagger_a, list_mo_local ):
    '''
    Create a matrix representation of the spin operators s_2 in the many-body basis 
    for a local set of molecular orbitals.

    Parameters
    ----------
    a_dagger_a : matrix representation of the a_dagger_a operator in the many-body basis.
 
    Returns
    -------
    s_2, s_plus, s_z :  matrix representation of the s_2, s_plus and s_z operators
                        in the many-body basis.
    '''
    dim_H = np.shape(a_dagger_a[0, 0].A)[0]
    s_plus = scipy.sparse.csr_matrix((dim_H, dim_H))
    s_z = scipy.sparse.csr_matrix((dim_H, dim_H))
    for p in list_mo_local:
        s_plus += a_dagger_a[2 * p, 2 * p + 1]
        s_z += (a_dagger_a[2 * p, 2 * p] - a_dagger_a[2 * p + 1, 2 * p + 1]) / 2.
    s_2 = s_plus @ s_plus.T + s_z @ s_z - s_z
    return s_2


def build_sAsB_coupling( a_dagger_a, list_mo_local_A, list_mo_local_B ):
    '''
    Create a matrix representation of the product of two local spin operators 
    s_A * s_B in the many-body basis. Each one being associated to a local set
    of molecular orbitals.

    Parameters
    ----------
    a_dagger_a : matrix representation of the a_dagger_a operator in the many-body basis.
 
    Returns
    -------
    sAsB_coupling :  matrix representation of s_A * s_B in the many-body basis.
    '''
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

    
# =============================================================================
#  FUNCTION TO GIVE ACCESS TO BASIC QUANTUM CHEMISTRY DATA FROM PSI4
# =============================================================================

def get_info_from_psi4( string_geometry,
                        basisset,
                        molecular_charge=0,
                        TELL_ME=False ):
    '''
    Function to realise an Hartree-Fock calculation on a given molecule and to 
    return all the associated information for futhrer correlated wavefunction
    calculation for QuantNBody.
    
    Parameters
    ----------
    S2 : matrix representation of the S2 operator in the many-body basis.
 
    Returns
    -------
    Projector_spin_subspace :  Projector on the targeted spin-subspace.
    '''
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
#  FUNCTION TO GENERATE AN OVERLPA BETWEEN TWO WAVEFUNCTION EXPRESSED 
#  IN TWO DIFFERENT MOLECULAR ORBITAL BASIS
# =============================================================================

def weight_det( C_B2_B1, occ_spinorb_Det1, occ_spinorb_Det2 ):
    
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
    
    return scipy.linalg.det( M ) 
 

def scalar_product_different_MO_basis( Psi_A_MOB1,
                                       Psi_B_MOB2,
                                       C_MOB1,
                                       C_MOB2,
                                       nbody_basis ):
    
    dim_H = len(nbody_basis)
    
    # Overlap matrix in the common basis
    S = scipy.linalg.inv( C_MOB1 @ C_MOB1.T )  
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


def prepare_vector_k_orital_rotation_fwith_active_space( Vec_k,
                                                         n_mo,
                                                         frozen_indices,
                                                         active_indices,
                                                         virtual_indices):
    """
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build tprepare the .....
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    """ 
    Vec_k = []
    for p in range(n_mo-1):
        for q in range(p+1, n_mo):
            if not( ( p in active_indices and q in active_indices) 
                 or ( p in frozen_indices and q in frozen_indices) 
                 or ( p in virtual_indices and q in virtual_indices)):
                Vec_k += [ 0 ]  
            
    return Vec_k

def transform_vec_to_skewmatrix_with_active_space( Vec_k,
                                                   n_mo,
                                                   frozen_indices,
                                                   active_indices ):
    """
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build the skew-matrix (Anti-Symmetric) generator matrix K for 
    the orbital rotations from a vector k.
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    """
    Skew_Matrix_K = np.zeros((n_mo, n_mo))
    ind_ij = 0
    for j in range(n_mo-1):
        for i in range(j+1, n_mo):
            Skew_Matrix_K[i, j] = Vec_k[ind_ij]
            Skew_Matrix_K[j, i] = - Vec_k[ind_ij]
            ind_ij += 1
    return Skew_Matrix_K

# =============================================================================
#  FUNCTION TO HELP THE VISUALIZATION OF MANY-BODY WAVE FUNCTIONS
# =============================================================================

def visualize_wft(WFT, nbody_basis, cutoff=0.005, atomic_orbitals=False):
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
        states += [nbody_basis[index]]

    list_sorted_index = np.flip(np.argsort(np.abs(coefficients)))

    return_string = f'\n\t{"-" * 11}\n\t Coeff.      N-body state\n\t{"-" * 7}     {"-" * 13}\n'
    for index in list_sorted_index[0:8]:
        state = states[index]

        if atomic_orbitals:
            ket = get_ket_in_atomic_orbitals(state, bra=False)
        else:
            ket = '|' + "".join([str(elem) for elem in state]) + '⟩'
        return_string += f'\t{coefficients[index]:+1.5f}\t{ket}\n'
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


# =============================================================================
# USEFUL TRANSFORMATIONS
# =============================================================================

def transform_1_2_body_tensors_in_new_basis(h_b1, g_b1, C):
    """
    Transform electronic integrals from an initial basis "B1" to a new basis "B2".
    The transformation is realized thanks to a passage matrix noted "C" linking
    both basis like

            | B2_l > = \\sum_{p} | B1_p >  C_{pl}

    with | B2_l > and | B1_p > are vectors of the basis B1 and B2 respectively

    Parameters
    ----------
    h_b1 : 1-electron integral given in basis B1
    g_b1 : 2-electron integral given in basis B1
    C    : Passage matrix

    Returns
    -------
    h_b2 : 1-electron integral given in basis B2
    g_b2 : 2-electron integral given in basis B2
    """
    h_b2 = np.einsum('pi,qj,pq->ij', C, C, h_b1, optimize=True)
    g_b2 = np.einsum('ap, bq, cr, ds, abcd -> pqrs', C, C, C, C, g_b1, optimize=True)

    return h_b2, g_b2


def householder_transformation(M):
    """
    Householder transformation transforming a squared matrix " M " into a
    block-diagonal matrix " M_BD " such that

                           M_BD = P M P

    where " P " represents the Householder transformation built from the
    vector "v" such that

                         P = Id - 2 * v.v^T

    NB : This returns a 2x2 block on left top corner

    Parameters
    ----------
    M :  Squared matrix to be transformed

    Returns
    -------
    P :  Transformation matrix
    v :  Householder vector

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
                           M_BD = H(V) M H(V)
    where ” H(V) ” represents the Householder transformation built from the
    matrix “V” such that
                        H(V) = Id - 2. * V(V^{T} V)^{-1}V^{T}
    NB : Depending on the size of the block needed (unchanged by the transformation),
         this returns a (2 x size)*(2 x size) block on left top corner
    ----------
    Article : F. Rotella, I. Zambettakis, Block Householder Transformation for Parallel QR Factorization,
              Applied Mathematics Letter 12 (1999) 29-34

    Parameters
    ----------
    M :  Square matrix to be transformed
    size : size of the block ( must be > 1)

    Returns
    -------
    P                 : Transformation matrix
    moore_penrose_inv : Moore Penrose inverse of Householder matrix
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
# ORBITAL OPTIMIZATION
# =============================================================================

@njit
def compute_energy_with_rdm(ONE_RDM,
                            TWO_RDM,
                            active_indices,
                            h_MO,
                            g_MO,
                            E_shift):
    """
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to compute the energy faster with only the knowledge of RDMs
    and 1 and 2-electron integrals
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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



def prepare_vector_k_orbital_rotation_fwith_active_space(n_mo,
                                                         frozen_indices,
                                                         active_indices,
                                                         virtual_indices):
    """
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build tprepare the .....
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    """ 
    Vec_k = []
    for p in range(n_mo-1):
        for q in range(p+1, n_mo):
            if not( ( p in frozen_indices and q in frozen_indices) 
                 or ( p in virtual_indices and q in virtual_indices)):
                Vec_k += [ 1e-4 ]  
            
    return Vec_k


def transform_vec_to_skewmatrix_with_active_space( Vec_k,
                                                   n_mo,
                                                   frozen_indices,
                                                   active_indices,
                                                   virtual_indices):
    """
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build the skew-matrix (Anti-Symmetric) generator matrix K for 
    the orbital rotations from a vector k.
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
    '''
    
    ''' 
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
                                      tolerance=1e-6,
                                      show_me=False,
                                      SAD_guess = True):
    '''
    
    ''' 
    n_mo = np.shape(h)[0]
    Vec_k = prepare_vector_k_orbital_rotation_fwith_active_space( n_mo,
                                                                    frozen_indices,
                                                                    active_indices,
                                                                    virtual_indices)
    
    if SAD_guess:
        MO_guess_energy, C_guess = np.linalg.eigh( h ) 
        h, g = transform_1_2_body_tensors_in_new_basis( h, g, C_guess )
    
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
                              # constraints=cons,
                              method  = method_name,
                              options = {'maxiter': max_iteration,
                                        'gtol'     : tolerance,
                                        'disp': show_me}  )
    Vec_k = f_min_OO['x']
    K_mat = transform_vec_to_skewmatrix_with_active_space(Vec_k,
                                                          n_mo,
                                                          frozen_indices,
                                                          active_indices,
                                                          virtual_indices)
    U_OO =  scipy.linalg.expm( - K_mat )  
    C_OO = C_ref @ U_OO 
    h_MO, g_MO = transform_1_2_body_tensors_in_new_basis( h, g, U_OO )
    E_new = compute_energy_with_rdm( one_rdm,
                                    two_rdm,
                                    active_indices, 
                                    h_MO,
                                    g_MO,
                                    E_rep_nuc )
    return C_OO, E_new, h_MO, g_MO


def filter_h_g_orb(Hess_OrbOrb,
                   Grad_Orb,
                   frozen_indices,
                   active_indices,
                   virtual_indices,
                   n_mo_optimized):
    """
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to filter the redundant terms in the Orbital gradient and Hessian
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build the MO hessian matrix and the MO gradient vector
    for the orbital Optimization in a STATE-AVERAGED way
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
def build_mo_gradient(n_mo_OPTIMIZED,
                      active_indices,
                      frozen_indices,
                      virtual_indices,
                      h_MO,
                      g_MO,
                      one_rdm,
                      two_rdm):
    """
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build a state specific MO  MO gradient vector
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    """
    Num_MO = np.shape(h_MO)[0]
    gradient_I = np.zeros((n_mo_OPTIMIZED * (n_mo_OPTIMIZED - 1) // 2, 1))
    F_I = build_generalized_fock_matrix(Num_MO,
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
            gradient_I[ind_pq] = 2. * (F_I[p, q] - F_I[q, p])

    return gradient_I


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
                                       OPT_OO_MAX_ITER,
                                       Grad_threshold,
                                       TELL_ME=True):
    """
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to realize an Orbital Optimization process
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build the skew-matrix (Anti-Symmetric) generator matrix K for 
    the orbital rotations from a vector k.
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build the generalized Fock matrix.
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build the generalized Fock matrix.
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build the inactive Fock matrix
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build the active Fock matrix.
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build the auxilliarry matrix.
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
    Function to obtain a "super index" pq 
    useful for orbital gradients and hessians
    """
    ini_int = n_mo-1 - q
    fin_int = n_mo-1
    counter = (fin_int-ini_int+1)*(ini_int+fin_int)//2
    ind_pq = counter + p - n_mo
    return ind_pq


@njit
def delta(index_1, index_2):
    """
    Function delta kronecker
    """
    d = 0.0
    if index_1 == index_2:
        d = 1.0
    return d

# =============================================================================
#  MISCELLANEOUS
# =============================================================================

def generate_h_chain_geometry(N_atoms, dist_HH):
    """
    A function to build a Hydrogen chain geometry (on the x-axis)

                   H - H - H - H

    N_atoms :: Total number of Hydrogen atoms
    dist_HH :: the interatomic distance
    """
    h_chain_geometry = 'H 0. 0. 0.'  # Define the position of the first atom
    for n in range(1, N_atoms):
        h_chain_geometry += '\nH {} 0. 0.'.format(n * dist_HH)

    return h_chain_geometry


def generate_h_ring_geometry(N_atoms, radius):
    """
    A function to build a Hydrogen ring geometry (in the x-y plane)
                         H - H
                        /     \\
                       H       H
                        \\     /
                         H - H
    N_atoms  :: Total number of Hydrogen atoms
    radius   :: Radius of the ring
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
    A function to build a Hydrogen ring geometry (in the x-y plane)
                         H - H )
                         H - H ) theta 
    radius   :: Radius of the ring
    """
    h4_geometry = """ H   {0}   {1}  0.
                      H   {0}  -{1}  0. 
                      H  -{0}   {1}  0.
                      H  -{0}  -{1}  0.
                      symmetry c1""".format( radius*np.cos(angle/2.), radius*np.sin(angle/2.) ) 
    return h4_geometry