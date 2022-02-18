import numpy as np
import math as m
import scipy
from numba import njit, prange
from itertools import combinations
from tqdm import tqdm


# =============================================================================
# CORE FUNCTIONS FOR THE BUILDING OF the "A_dagger A" OPERATOR
# =============================================================================

def Build_NBody_Basis(N_MO, N_elec, S_z_cleaning=False):
    """
    Create a many-body basis formed by a list of slater-determinants
    (i.e. their occupation number)

    Parameters
    ----------
    N_MO         :  Number of molecular orbitals
    N_elec       :  Number of electrons
    S_z_cleaning :  Option if we want to get read of the S_z != 0 states (default is False)

    Returns
    -------
    NBody_Basis :  List of many-body states (occupation number states) in the basis (occupation number vectors)
    """
    # Building the N-electron many-body basis
    NBody_Basis = []
    for combination in combinations(range(2 * N_MO), N_elec):
        fockstate = [0 for i in range(2 * N_MO)]
        for index in list(combination):
            fockstate[index] += 1
        NBody_Basis += [fockstate]

        # In case we want to get rid of states with S_z != 0
    if S_z_cleaning:
        NBody_Basis_cleaned = NBody_Basis.copy()
        for i in range(np.shape(NBody_Basis)[0]):
            S_z = Check_Sz(NBody_Basis[i])
            if S_z != 0:
                NBody_Basis_cleaned.remove(NBody_Basis[i])
        NBody_Basis = NBody_Basis_cleaned

    return NBody_Basis


def Check_Sz(ref_state):
    """
    Return the value fo the S_z operator for a unique slater determinant

    Parameters
    ----------
    ref_state :  Slater determinant (list of occupation numbers)

    Returns
    -------
    S_z_slater_determinant : value of S_z for the given slater determinant
    """
    S_z_slater_determinant = 0
    for elem in range(len(ref_state)):
        if elem % 2 == 0:
            S_z_slater_determinant += + 1 * ref_state[elem] / 2
        else:
            S_z_slater_determinant += - 1 * ref_state[elem] / 2

    return S_z_slater_determinant


def build_operator_a_dagger_a(NBody_Basis):
    """
    Create a matrix representation of the a_dagger_a operator
    in the many-body basis

    Parameters
    ----------
    NBody_Basis :  List of many-body states (occupation number states) (occupation number states)

    Returns
    -------
    a_dagger_a :  Matrix representation of the a_dagger_a operator

    """
    # Dimensions of problem
    dim_H = len(NBody_Basis)
    N_MO = len(NBody_Basis[0]) // 2
    Mapping_kappa_ = Build_mapping(np.array(NBody_Basis))

    a_dagger_a = np.zeros((2 * N_MO, 2 * N_MO), dtype=object)
    for p in range(2 * N_MO):
        for q in range(p, 2 * N_MO):
            a_dagger_a[p, q] = scipy.sparse.lil_matrix((dim_H, dim_H))
            a_dagger_a[q, p] = scipy.sparse.lil_matrix((dim_H, dim_H))

    for MO_q in (range(N_MO)):
        for MO_p in range(MO_q, N_MO):
            for kappa in range(dim_H):
                ref_state = NBody_Basis[kappa]

                # Single excitation : spin alpha -- alpha
                p, q = 2 * MO_p, 2 * MO_q
                if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
                    pass
                elif ref_state[q] == 1:
                    kappa_, p1, p2 = Build_final_state_ad_a(np.array(ref_state), p, q, Mapping_kappa_)
                    a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] = p1 * p2

                    # Single excitation : spin beta -- beta
                p, q = 2 * MO_p + 1, 2 * MO_q + 1
                if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
                    pass
                elif ref_state[q] == 1:
                    kappa_, p1, p2 = Build_final_state_ad_a(np.array(ref_state), p, q, Mapping_kappa_)
                    a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] = p1 * p2

                if MO_p == MO_q:  # <=== Necessary to build the Spins operator but not really for Hamiltonians

                    # Single excitation : spin beta -- alpha
                    p, q = 2 * MO_p + 1, 2 * MO_p
                    if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
                        pass
                    elif ref_state[q] == 1:
                        kappa_, p1, p2 = Build_final_state_ad_a(np.array(ref_state), p, q, Mapping_kappa_)
                        a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] = p1 * p2

                        # Single excitation : spin alpha -- beta
                    p, q = 2 * MO_p, 2 * MO_p + 1

                    if p != q and (ref_state[q] == 0 or ref_state[p] == 1):
                        pass
                    elif ref_state[q] == 1:
                        kappa_, p1, p2 = Build_final_state_ad_a(np.array(ref_state), p, q, Mapping_kappa_)
                        a_dagger_a[p, q][kappa_, kappa] = a_dagger_a[q, p][kappa, kappa_] = p1 * p2

    print()
    print('\t ===========================================')
    print('\t ====  The matrix form of a^a is built  ====')
    print('\t ===========================================')

    return a_dagger_a


@njit
def Build_mapping(NBody_Basis):
    """
    Function to create a unique mapping between a kappa vector and an occupation
    number state.

    Parameters
    ----------
    NBody_Basis :  Many-

    Returns
    -------
    Mapping_kappa : List of unique values associated to each kappa
    """
    Mapping_kappa = np.zeros(10 ** 9, dtype=np.int32)
    Num_digits = np.shape(NBody_Basis)[1]
    dim_H = np.shape(NBody_Basis)[0]
    for kappa in range(dim_H):
        ref_state = NBody_Basis[kappa]
        number = 0
        for digit in range(Num_digits):
            number += ref_state[digit] * 2 ** (Num_digits - digit - 1)
        Mapping_kappa[number] = kappa

    return Mapping_kappa


@njit
def Make_integer_out_of_bit_vector(ref_state):
    """
    Function to translate a slater determinant into an unique integer

    Parameters
    ----------
    ref_state : Reference slater determinant to turn out into an integer

    Returns
    -------
    number : unique integer  refering to the slater determinant
    """
    number = 0
    for digit in range(len(ref_state)):
        number += ref_state[digit] * 2 ** (len(ref_state) - digit - 1)

    return number


@njit
def New_state_after_SQ_fermi_op(type_of_op, index_mode, ref_fockstate):
    """

    Parameters
    ----------
    type_of_op    :  type of operator to apply (creation of anihilation)
    index_mode    :  index of the second quantized mode to occupy/empty
    ref_fockstate :  initial state to be transformed

    Returns
    -------
    new_fockstate :  Resulting occupation number form of the transformed state
    coeff_phase   :  Phase attached to the resulting state

    """
    new_fockstate = ref_fockstate.copy()
    coeff_phase = 1
    if type_of_op == 'a':
        new_fockstate[index_mode] += -1
        if index_mode > 0:
            coeff_phase = (-1.) ** np.sum(ref_fockstate[0:index_mode])
    elif type_of_op == 'a^':
        new_fockstate[index_mode] += 1
        if index_mode > 0:
            coeff_phase = (-1.) ** np.sum(ref_fockstate[0:index_mode])

    return new_fockstate, coeff_phase


@njit
def Build_final_state_ad_a(ref_state, p, q, Mapping_kappa_):
    state_one, p1 = New_state_after_SQ_fermi_op('a', q, ref_state)
    state_two, p2 = New_state_after_SQ_fermi_op('a^', p, state_one)
    kappa_ = Mapping_kappa_[Make_integer_out_of_bit_vector(state_two)]

    return kappa_, p1, p2


def My_State(Slater_determinant, NBody_Basis):
    """
    Translate a Slater determinant (occupation numbe list) into a many-body
    state referenced into a given Many-body basis.

    Parameters
    ----------
    NBody_Basis   : List of many-body states (occupation number states)

    Returns
    -------
    State :  The slater determinant referenced in the many-body basis
    """
    kappa = NBody_Basis.index(Slater_determinant)
    State = np.zeros(np.shape(NBody_Basis)[0])
    State[kappa] = 1.

    return State


# =============================================================================
#  MANY-BODY HAMILTONIANS (FERMI HUBBARD AND QUANTUM CHEMISTRY)
# =============================================================================

def Build_Hamiltonian_Quantum_Chemistry(h_, g_, NBody_Basis, a_dagger_a, S_2=None, S_2_target=None, penalty=100):
    """
    Create a matrix representation of the electornic strucutre Hamiltonian in any
    extended many-body basis

    Parameters
    ----------
    h_          :  One-body integrals
    g_          :  TODO: What is parameter name, maybe Two-body integrals
    NBody_Basis :  List of many-body states (occupation number states)
    a_dagger_a  :  Matrix representation of the a_dagger_a operator
    S_2         :  Matrix representation of the S_2 operator (default is None)
    S_2_target  :  Value of the S_2 mean value we want to target (default is None)
    penalty     :  Value of the penalty term for state not respecting the spin symmetry (default is 100).

    Returns
    -------
    H_Chemistry :  Matrix representation of the electornic structure Hamiltonian

    """
    # Dimension of the problem 
    dim_H = len(NBody_Basis)
    N_MO = np.shape(h_)[0]

    # Building the spin-preserving one-body excitation operator  
    E_ = np.empty((2 * N_MO, 2 * N_MO), dtype=object)
    e_ = np.empty((2 * N_MO, 2 * N_MO, 2 * N_MO, 2 * N_MO), dtype=object)
    for p in range(N_MO):
        for q in range(N_MO):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]

    for p in range(N_MO):
        for q in range(N_MO):
            for r in range(N_MO):
                for s in range(N_MO):
                    e_[p, q, r, s] = E_[p, q] @ E_[r, s]
                    if q == r:
                        e_[p, q, r, s] += - E_[p, s]

                    # Building the N-electron electronic structure hamiltonian
    H_Chemistry = scipy.sparse.csr_matrix((dim_H, dim_H))
    for p in range(N_MO):
        for q in range(N_MO):
            H_Chemistry += E_[p, q] * h_[p, q]
            for r in range(N_MO):
                for s in range(N_MO):
                    H_Chemistry += e_[p, q, r, s] * g_[p, q, r, s] / 2.

                    # Reminder : S_2 = S(S+1) and the total spin multiplicity is 2S+1
    # with S = the number of unpaired electrons x 1/2
    # singlet    =>  S=0    and  S_2=0
    # doublet    =>  S=1/2  and  S_2=3/4
    # triplet    =>  S=1    and  S_2=2
    # quadruplet =>  S=3/2  and  S_2=15/4
    # quintet    =>  S=2    and  S_2=6
    if S_2 is not None and S_2_target is not None:
        S_2_minus_target = S_2 - S_2_target * np.eye(dim_H)
        H_Chemistry += S_2_minus_target @ S_2_minus_target * penalty

    return H_Chemistry


def Build_Hamiltonian_Fermi_Hubbard(h_, U_, NBody_Basis, a_dagger_a, S_2=None, S_2_target=None, penalty=100):
    """
    Create a matrix representation of the Fermi-Hubbard Hamiltonian in any
    extended many-body basis.

    Parameters
    ----------
    h_          :  One-body integrals
    U_          :  Two-body integrals
    NBody_Basis :  List of many-body states (occupation number states)
    a_dagger_a  :  Matrix representation of the a_dagger_a operator
    S_2         :  Matrix representation of the S_2 operator (default is None)
    S_2_target  :  Value of the S_2 mean value we want to target (default is None)
    penalty     :  Value of the penalty term for state not respecting the spin symmetry (default is 100).

    Returns
    -------
    H_Fermi_Hubbard :  Matrix representation of the Fermi-Hubbard Hamiltonian

    """
    # # Dimension of the problem 
    dim_H = len(NBody_Basis)
    N_MO = np.shape(h_)[0]

    # Building the N-electron Fermi-Hubbard matrix hamiltonian (Sparse)
    H_Fermi_Hubbard = scipy.sparse.csr_matrix((dim_H, dim_H))
    for p in tqdm(range(N_MO)):
        for q in range(N_MO):
            H_Fermi_Hubbard += (a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]) * h_[p, q]
            for r in range(N_MO):
                for s in range(N_MO):
                    H_Fermi_Hubbard += a_dagger_a[2 * p, 2 * q] @ a_dagger_a[2 * r + 1, 2 * s + 1] * U_[p, q, r, s]

    # Reminder : S_2 = S(S+1) and the total  spin multiplicity is 2S+1 
    # with S = the number of unpaired electrons x 1/2 
    # singlet    =>  S=0    and  S_2=0 
    # doublet    =>  S=1/2  and  S_2=3/4
    # triplet    =>  S=1    and  S_2=2
    # quadruplet =>  S=3/2  and  S_2=15/4
    # quintet    =>  S=2    and  S_2=6
    if S_2 is not None and S_2_target is not None:
        S_2_minus_target = S_2 - S_2_target * np.eye(dim_H)
        H_Fermi_Hubbard += S_2_minus_target @ S_2_minus_target * penalty

    return H_Fermi_Hubbard


def FH_get_active_space_integrals(h_, U_, frozen_indices=None, active_indices=None):
    """
        >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        Restricts a Fermi-Hubard at a spatial orbital level to an active space
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
        >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        """
    # Determine core Energy from frozen MOs
    Core_energy = 0
    for i in frozen_indices:
        Core_energy += 2 * h_[i, i]
        for j in frozen_indices:
            Core_energy += U_[i, i, j, j]

            # Modified one-electron integrals
    h_act = h_.copy()
    for t in active_indices:
        for u in active_indices:
            for i in frozen_indices:
                h_act[t, u] += U_[i, i, t, u]

    return (Core_energy,
            h_act[np.ix_(active_indices, active_indices)],
            U_[np.ix_(active_indices, active_indices, active_indices, active_indices)])


def QC_get_active_space_integrals(one_body_integrals, two_body_integrals, occupied_indices=None, active_indices=None):
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
            core_constant += (2 * two_body_integrals[i, j, i, j] - two_body_integrals[i, j, j, i])

    # Modified one electron integrals
    one_body_integrals_new = np.copy(one_body_integrals)
    for u in active_indices:
        for v in active_indices:
            for i in occupied_indices:
                one_body_integrals_new[u, v] += (2 * two_body_integrals[i, i, u, v]
                                                 - two_body_integrals[i, u, v, i])

    # Restrict integral ranges and change M appropriately
    return (core_constant,
            one_body_integrals_new[np.ix_(active_indices, active_indices)],
            two_body_integrals[np.ix_(active_indices, active_indices, active_indices, active_indices)])


# =============================================================================
#  DIFFERENT TYPES OF SPIN OPERATORS
# =============================================================================

def Build_S2_SZ_Splus_operator(a_dagger_a):
    """
    Create a matrix representation of the spin operators S_2, S_z and S_plus
    in the many-body basis.

    Parameters
    ----------
    a_dagger_a : matrix representation of the a_dagger_a operator in the many-body basis.

    Returns
    -------
    S_2, S_plus, S_z :  matrix representation of the S_2, S_plus and S_z operators
                        in the many-body basis.
    """
    N_MO = np.shape(a_dagger_a)[0] // 2
    dim_H = np.shape(a_dagger_a[0, 0].A)[0]
    S_plus = scipy.sparse.csr_matrix((dim_H, dim_H))
    S_z = scipy.sparse.csr_matrix((dim_H, dim_H))
    for p in range(N_MO):
        S_plus += a_dagger_a[2 * p, 2 * p + 1]
        S_z += (a_dagger_a[2 * p, 2 * p] - a_dagger_a[2 * p + 1, 2 * p + 1]) / 2.

    S_2 = S_plus @ S_plus.T + S_z @ S_z - S_z

    return S_2, S_plus, S_z


# =============================================================================
#  DIFFERENT TYPES OF REDUCED DENSITY-MATRICES
# =============================================================================

def Build_One_RDM_alpha(WFT, a_dagger_a):
    """
    Create a spin-alpha 1 RDM out of a given wavefunction

    Parameters
    ----------
    WFT        :  Wavefunction for which we want to build the 1-RDM
    a_dagger_a :  Matrix representation of the a_dagger_a operator

    Returns
    -------
    One_RDM_alpha : spin-alpha 1-RDM

    """
    N_MO = np.shape(a_dagger_a)[0] // 2
    one_rdm_alpha = np.zeros((N_MO, N_MO))
    for p in range(N_MO):
        for q in range(N_MO):
            one_rdm_alpha[p, q] = WFT.T @ a_dagger_a[2 * p, 2 * q] @ WFT
    return one_rdm_alpha


def Build_One_RDM_beta(WFT, a_dagger_a):
    """
    Create a spin-beta 1 RDM out of a given wavefunction

    Parameters
    ----------
    WFT        :  Wavefunction for which we want to build the 1-RDM
    a_dagger_a :  Matrix representation of the a_dagger_a operator

    Returns
    -------
    One_RDM_alpha : Spin-beta 1-RDM

    """
    N_MO = np.shape(a_dagger_a)[0] // 2
    one_rdm_beta = np.zeros((N_MO, N_MO))
    for p in range(N_MO):
        for q in range(N_MO):
            one_rdm_beta[p, q] = WFT.T @ a_dagger_a[2 * p + 1, 2 * q + 1] @ WFT
    return one_rdm_beta


def Build_One_RDM_spin_free(WFT, a_dagger_a):
    """
    Create a spin-free 1 RDM out of a given wavefunction

    Parameters
    ----------
    WFT        :  Wavefunction for which we want to build the 1-RDM
    a_dagger_a :  Matrix representation of the a_dagger_a operator

    Returns
    -------
    One_RDM_alpha : Spin-free 1-RDM

    """
    N_MO = np.shape(a_dagger_a)[0] // 2
    one_rdm = np.zeros((N_MO, N_MO))
    for p in range(N_MO):
        for q in range(N_MO):
            E_pq = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]
            one_rdm[p, q] = WFT.T @ E_pq @ WFT
    return one_rdm


def Build_two_RDM_FH(WFT, a_dagger_a):
    """
    Create a spin-free 1 RDM out of a given wavefunction

    Parameters
    ----------
    WFT        :  Wavefunction for which we want to build the 1-RDM
    a_dagger_a :  Matrix representation of the a_dagger_a operator

    Returns
    -------
    One_RDM_alpha : Spin-free 1-RDM

    """
    N_MO = np.shape(a_dagger_a)[0] // 2
    two_RDM_FH = np.zeros((N_MO, N_MO, N_MO, N_MO))
    for p in range(N_MO):
        for q in range(N_MO):
            for r in range(N_MO):
                for s in range(N_MO):
                    two_RDM_FH[p, q, r, s] += WFT.T @ a_dagger_a[2 * p, 2 * q] @ a_dagger_a[2 * r + 1, 2 * s + 1] @ WFT

    return two_RDM_FH


def Build_two_RDM_spin_free(WFT, a_dagger_a):
    """
    Create a spin-free 2 RDM out of a given wavefunction

    Parameters
    ----------
    WFT        :  Wavefunction for which we want to build the 1-RDM
    a_dagger_a :  Matrix representation of the a_dagger_a operator

    Returns
    -------
    One_RDM_alpha : Spin-free 1-RDM

    """
    N_MO = np.shape(a_dagger_a)[0] // 2
    two_RDM = np.zeros((N_MO, N_MO, N_MO, N_MO))
    E_ = np.empty((2 * N_MO, 2 * N_MO), dtype=object)
    for p in range(N_MO):
        for q in range(N_MO):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]

    for p in range(N_MO):
        for q in range(N_MO):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]
            for r in range(N_MO):
                for s in range(N_MO):
                    E_[r, s] = a_dagger_a[2 * r, 2 * s] + a_dagger_a[2 * r + 1, 2 * s + 1]
                    two_RDM[p, q, r, s] = WFT.T @ E_[p, q] @ E_[r, s] @ WFT
                    if q == r:
                        two_RDM[p, q, r, s] += - WFT.T @ E_[p, s] @ WFT

    return two_RDM


def Build_One_and_Two_RDM_spin_free(WFT, a_dagger_a):
    """
    Create a spin-free 2 RDM out of a given wavefunction

    Parameters
    ----------
    WFT        :  Wavefunction for which we want to build the 1-RDM
    a_dagger_a :  Matrix representation of the a_dagger_a operator

    Returns
    -------
    One_RDM_alpha : Spin-free 1-RDM

    """
    N_MO = np.shape(a_dagger_a)[0] // 2
    one_rdm = np.zeros((N_MO, N_MO))
    two_RDM = np.zeros((N_MO, N_MO, N_MO, N_MO))
    E_ = np.empty((2 * N_MO, 2 * N_MO), dtype=object)
    for p in range(N_MO):
        for q in range(N_MO):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]

    for p in range(N_MO):
        for q in range(N_MO):
            E_[p, q] = a_dagger_a[2 * p, 2 * q] + a_dagger_a[2 * p + 1, 2 * q + 1]
            one_rdm[p, q] = WFT.T @ E_[p, q] @ WFT
            for r in range(N_MO):
                for s in range(N_MO):
                    E_[r, s] = a_dagger_a[2 * r, 2 * s] + a_dagger_a[2 * r + 1, 2 * s + 1]
                    two_RDM[p, q, r, s] = WFT.T @ E_[p, q] @ E_[r, s] @ WFT
                    if q == r:
                        two_RDM[p, q, r, s] += - WFT.T @ E_[p, s] @ WFT

    return one_rdm, two_RDM


# =============================================================================
#  FUNCTION TO HELP THE VISUALIZATION OF MANY-BODY WAVEFUNCTIONS
# =============================================================================

def Visualize_WFT(WFT, NBody_Basis, cutoff=0.005):
    """
    Print the decomposition of a given input wavefunction in a many-body basis.
    ----------
    WFT            : Reference wavefunction
    NBody_Basis    : List of many-body states (occupation number states)
    cutoff         : Cuoff for the amplitudes retained (default is 0.005)

    Returns
    -------
    Printing in the terminal the wavefunction

    """
    list_indx = np.where(abs(WFT) > cutoff)[0]

    States = []
    Coeffs = []
    for indx in list_indx:
        Coeffs += [WFT[indx]]
        States += [NBody_Basis[indx]]

    list_sorted_indx = np.flip(np.argsort(np.abs(Coeffs)))

    print()
    print('\t ----------- ')
    print('\t Coeff.      N-body state')
    print('\t -------     -------------')
    for indx in list_sorted_indx[0:8]:
        sign_ = '+'
        if abs(Coeffs[indx]) != Coeffs[indx]:
            sign_ = '-'
        print('\t', sign_, '{:1.5f}'.format(abs(Coeffs[indx])),
              '\t|{}⟩'.format(' '.join([str(elem) for elem in States[indx]]).replace(" ", "")))

    return


# =============================================================================
# USEFUL TRANSFORMATIONS 
# =============================================================================


def Transform_1_2_body_tensors_in_new_basis(h_B1, g_B1, C):
    """
    Transform electronic integrals from an initial basis "B1" to a new basis "B2".
    The transformation is realized thanks to a passage matrix noted "C" linking
    both basis like

            | B2_l > = \sum_{p} | B1_p >  C_{pl}

    with | B2_l > and | B1_p > are vectors of the basis B1 and B2 respectively

    Parameters
    ----------
    h_B1 : 1-electron integral given in basis B1
    g_B1 : 2-electron integral given in basis B1
    C    : Passage matrix

    Returns
    -------
    h_B2 : 1-electron integral given in basis B2
    g_B2 : 2-electron integral given in basis B2
    """
    h_B2 = np.einsum('pi,qj,pq->ij', C, C, h_B1)
    g_B2 = np.einsum('ap, bq, cr, ds, abcd -> pqrs', C, C, C, C, g_B1)

    return h_B2, g_B2


def Householder_transformation(M):
    """
    Householder transformation transforming a squarred matrix " M " into a
    block-diagonal matrix " M_BD " such that

                           M_BD = P M P

    where " P " represents the Householrder transfomration built from the
    vector "v" such that

                         P = Id - 2 * v.v^T

    NB : This returns a 2x2 block on left top corner

    Parameters
    ----------
    M :  Squarred matrix to be transformed

    Returns
    -------
    P :  Transformation matrix
    v :  Householder vector

    """
    N = np.shape(M)[0]
    # Build the Housholder vector "H_vector" 
    alpha = - np.sign(M[1, 0]) * sum(M[j, 0] ** 2. for j in range(1, N)) ** 0.5
    r = (0.5 * (alpha ** 2. - alpha * M[1, 0])) ** 0.5

    # print("HH param transfomration",M,r,alpha)
    vector = np.zeros((N, 1))
    vector[1] = (M[1, 0] - alpha) / (2.0 * r)
    for j in range(2, N):
        vector[j] = M[j, 0] / (2.0 * r)

    # Building the transformation matrix "P" 
    P = np.eye(N) - 2 * vector @ vector.T

    return P, vector


def Block_householder_transformation(M, Block_size):
    """
    Block Householder transformation transforming a squarred matrix ” M ” into a
    block-diagonal matrix ” M_BD ” such that
                           M_BD = P M P
    where ” P ” represents the Householrder transfomration built from the
    vector “v” such that
                         !!!!!!P = Id - 2 * v.v^T                                TO BE CHANGED !!!
    NB : This returns a 2x2 block on left top corner
    Parameters
    ----------
    M :  Squarred matrix to be transformed
    Returns
    -------
    P :  Transformation matrix
    v :  Householder vector
    """
    Block_size = Block_size // 2
    N = np.shape(M)[0]

    """ WILL BE WITH THE INVERSE SIGN OF X """
    A1 = M[:Block_size, Block_size:2 * Block_size]
    A1_inv = np.linalg.inv(A1)
    A2 = M[:Block_size, 2 * Block_size:]
    A2A1_inv_tr = np.zeros((Block_size, N - 2 * Block_size))
    for i in range(N - 2 * Block_size):
        for j in range(Block_size):
            for k in range(Block_size):
                A2A1_inv_tr[j, i] = A2A1_inv_tr[j, i] + A2[k, i] * A1_inv[j, k]

    # A2A1_inv = np.transpose(A2A1_inv_tr) 
    A3 = np.eye(Block_size) + A2A1_inv_tr @ A2A1_inv_tr.T
    w, v = np.linalg.eig(A3)
    eigval = np.diag(w)
    Xd = v.T @ eigval ** 0.5 @ v

    X = A1 @ Xd
    Y = A1 + X
    V_tr = np.block([np.zeros((Block_size, Block_size)), Y, M[:Block_size, 2 * Block_size:]])

    V_trV = V_tr @ V_tr.T
    V_trV_inv = np.linalg.inv(V_trV)
    BH = np.eye(N) - 2. * V_tr.T @ V_trV_inv @ V_tr

    return BH


# =============================================================================
#  MISCELENAOUS
# =============================================================================

def Build_MO_1_and_2_RDMs(Psi_A, active_indices, N_MO, E_precomputed, e_precomputed):
    """
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Function to build the MO 1/2-ELECTRON DENSITY MATRICES from a 
    reference wavefunction expressed in the computational basis
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    """
    ONE_RDM_A = np.zeros((N_MO, N_MO))
    TWO_RDM_A = np.zeros((N_MO, N_MO, N_MO, N_MO))
    first_act_index = active_indices[0]
    # Creating RDMs elements only within the frozen space
    if first_act_index > 0:
        for I in range(first_act_index):
            for J in range(first_act_index):
                ONE_RDM_A[I, J] = 2. * delta(I, J)
                for K in range(first_act_index):
                    for L in range(first_act_index):
                        # State A
                        TWO_RDM_A[I, J, K, L] = 4. * delta(I, J) * delta(K, L) - 2. * delta(I, L) * delta(J, K)

                        # Creating RDMs elements in the the active/frozen spaces
    for P in active_indices:
        for Q in active_indices:
            # Shifting the indices
            P_ = P - first_act_index
            Q_ = Q - first_act_index
            # 1-RDM elements only within the active space
            # State A
            ONE_RDM_A[P, Q] = (np.conj(Psi_A.T) @ E_precomputed[P_, Q_] @ Psi_A).real

            # 2-RDM elements only within the active space
            for R in active_indices:
                for S in active_indices:
                    # Shifting the indices
                    R_ = R - first_act_index
                    S_ = S - first_act_index
                    # State A
                    TWO_RDM_A[P, Q, R, S] = (np.conj(Psi_A.T) @ e_precomputed[P_, Q_, R_, S_] @ Psi_A).real

            if first_act_index > 0:
                # 2-RDM elements between the active and frozen spaces
                for I in range(first_act_index):
                    for J in range(first_act_index):
                        # State A
                        TWO_RDM_A[I, J, P, Q] = TWO_RDM_A[P, Q, I, J] = 2. * delta(I, J) * ONE_RDM_A[P, Q]
                        TWO_RDM_A[P, I, J, Q] = TWO_RDM_A[J, Q, P, I] = - delta(I, J) * ONE_RDM_A[P, Q]

    return ONE_RDM_A, TWO_RDM_A


def Generate_H_chain_geometry(N_atoms, dist_HH):
    """
    A function to build a Hydrogen chain geometry (on the x-axis)

                   H - H - H - H

    N_atoms :: Total number of Hydrogen atoms
    dist_HH :: the interatomic distance
    """
    H_chain_geometry = 'H 0. 0. 0.'  # Define the position of the first atom
    for n in range(1, N_atoms):
        H_chain_geometry += '\nH {} 0. 0.'.format(n * dist_HH)

    return H_chain_geometry


def Generate_H_ring_geometry(N_atoms, radius):
    """
    A function to build a Hydrogen ring geometry (in the x-y plane)
                         H - H
                        /     \
                       H       H
                        \     /
                         H - H
    N_atoms  :: Total number of Hydrogen atoms
    radius   :: Radius of the ring
    """
    theta_HH = 2 * m.pi / N_atoms  # Angle separating two consecutive H atoms (homogene distribution)
    theta_ini = 0.0
    H_ring_geometry = '\nH {:.16f} {:.16f} 0.'.format(radius * m.cos(theta_ini), radius * m.sin(
        theta_ini))  # 'H {:.16f} 0. 0.'.format( radius )  # Define the position of the first atom
    for n in range(1, N_atoms):
        angle_H = theta_ini + theta_HH * n
        H_ring_geometry += '\nH {:.16f} {:.16f} 0.'.format(radius * m.cos(angle_H), radius * m.sin(angle_H))

    return H_ring_geometry


def delta(index_1, index_2):
    """
    Function delta kronecker
    """
    d = 0.0
    if index_1 == index_2:
        d = 1.0
    return d
