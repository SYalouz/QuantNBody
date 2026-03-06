import scipy
import numpy as np
from itertools import combinations_with_replacement
from numba import njit, prange
import matplotlib.pyplot as plt
import scipy.sparse

# =============================================================================
# CORE FUNCTIONS FOR THE BUILDING OF the "A_dagger A" OPERATOR
# =============================================================================

def build_nbody_basis( n_mode, n_boson ):
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
    for combination in combinations_with_replacement( range(n_mode), n_boson ):
        fock_state = [ 0 for i in range(n_mode) ]
        for index in list(combination):
            fock_state[ index ] += 1
        nbody_basis += [ fock_state ]

    return np.array(nbody_basis)  # If pybind11 is used it is better to set dtype=np.int8

def build_mapping_dict(nbody_basis):
    dim_H = np.shape(nbody_basis)[0]
    mapping_kappa = {}
    
    for kappa in range(dim_H):
        ref_state = nbody_basis[kappa]
        bit_string = ''.join(str(int(x)) for x in ref_state)
        mapping_kappa[bit_string] = kappa  
    
    return mapping_kappa


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







    return kappa_, p1, p2

def build_final_state_ad_a(ref_state, p, q, mapping_kappa):
    state_one, p1 = new_state_after_sq_boson_op('a', q, ref_state)
    state_two, p2 = new_state_after_sq_boson_op('a^', p, state_one)
    
    bit_string = ''.join(str(x) for x in state_two)

    kappa_ = mapping_kappa[bit_string]

    return kappa_, p1, p2          



def build_operator_a_dagger_a(nbodybasis, silent=True):
    # Dimensions of problem
    dim_H  = len(nbodybasis)
    n_mode = len(nbodybasis[0])
    n_boson= np.sum(nbodybasis[0])
    mapping_kappa = build_mapping_dict(nbodybasis) 

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
def plot_wavefunctions(WFT, nbody_basis, list_states=[0], probability=False, cutoff=0.005, label_props=["x-large", 16, 16, 16]):
    import matplotlib.pyplot as plt

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
    list_states = list_states[::-1]
    indices_by_state = [np.where(np.abs(WFT[:, i]) > cutoff)[
        0] for i in list_states]

    height = [len(indices) for indices in indices_by_state]
    gridspecs = {'height_ratios': height} if N > 1 else {'height_ratios': None}

    fig, axs = plt.subplots(N, 1, sharex=True, dpi=150, gridspec_kw=gridspecs)

    # Loop through each state to plot
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
            L_label.append(
                '|' + ",".join([str(elem) for elem in state]) + '⟩'
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

@njit
def delta(index_1, index_2):
    """
    Function delta kronecker
    """
    d = 0.0
    if index_1 == index_2:
        d = 1.0
    return d
