import scipy
import numpy as np
from itertools import combinations
from numba import njit
from .tools import build_mapping, build_final_state_ad_a


def build_operator_a_dagger_a(nbody_basis, silent=False):
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
    n_mo = nbody_basis.shape[1] // 2
    a_dagger_a = np.zeros((2 * n_mo, 2 * n_mo), dtype=object)
    mapping_kappa = build_mapping(nbody_basis)
    n_e = np.sum(nbody_basis[0])
    for q in (range(2 * n_mo)):
        for p in range(q, 2 * n_mo):
            # 1. counting operator or p==q
            x_list, y_list, value_list = calculate_sparse_elements(p, q, nbody_basis, mapping_kappa, dim_H, n_e, n_mo)
            if len(x_list) == 0:
                temp1 = scipy.sparse.csr_matrix((dim_H, dim_H))  # if in cpp , dtype=np.int8
            else:
                temp1 = scipy.sparse.csr_matrix((value_list, (y_list, x_list)), shape=(dim_H, dim_H))  # dtype=np.int8
            a_dagger_a[p, q] = temp1
            a_dagger_a[q, p] = temp1.T

    if not silent:
        print()
        print('\t ===========================================')
        print('\t ====  The matrix form of a^a is built  ====')
        print('\t ===========================================')

    return a_dagger_a


@njit
def binomial(n, k):
    ret = 1
    if k > n - k:
        k = n - k
    for i in range(k):
        ret *= n - i
        ret /= i + 1
    return int(ret)


@njit
def calculate_sparse_elements(p, q, nbody_basis, mapping_kappa, dim_H, n_electron, n_mo):
    """
    This was prototype for c++ implementation of this function
    Parameters
    ----------
    p
    q
    nbody_basis
    mapping_kappa
    dim_H
    n_electron
    n_mo


    Returns
    -------
    """
    if p == q:
        i = 0
        sparse_num = int(dim_H * n_electron / (n_mo * 2))
        x_list = [0] * sparse_num
        y_list = [0] * sparse_num
        value_list = [0] * sparse_num
        for kappa in range(dim_H):
            ref_state = nbody_basis[kappa]
            # anything is happening only if ref_state[q] != 0
            if ref_state[q] == 0:
                continue
            x_list[i] = kappa
            y_list[i] = kappa
            value_list[i] = 1
            i += 1
    elif (p // 2 == q // 2) or ((p - q) % 2 == 0):
        # In the first case these are alpha beta excitations in same MO
        # In the second case spins of spin orbitals are the same and These are alpha beta excitations in same MO
        i = 0
        sparse_num = binomial(2 * n_mo - 2, n_electron - 1)
        x_list = [0] * sparse_num
        y_list = [0] * sparse_num
        value_list = [0] * sparse_num
        for kappa in range(dim_H):
            ref_state = nbody_basis[kappa]
            # anything is happening only if ref_state[q] != 0
            if ref_state[q] == 0 or ref_state[p] == 1:
                continue
            kappa2, p1, p2 = build_final_state_ad_a(ref_state, p, q, mapping_kappa)
            p1p2 = p1 * p2
            x_list[i] = kappa
            y_list[i] = kappa2
            value_list[i] = p1p2
            i += 1
    else:
        return [0], [0], [0]
    return x_list, y_list, value_list
