import tools_file_bosons
import numpy as np
from itertools import combinations
import scipy.sparse


# def build_nbody_basis(n_mo, N_electron, S_z_cleaning=False):
#     """
#     Create a many-body basis formed by a list of slater-determinants
#     (i.e. their occupation number)

#     Parameters
#     ----------
#     n_mo         :  Number of molecular orbitals
#     N_electron   :  Number of electrons
#     S_z_cleaning :  Option if we want to get read of the s_z != 0 states (default is False)

#     Returns
#     -------
#     nbody_basis :  List of many-body states (occupation number states) in the basis (occupation number vectors)
#     """
#     # Building the N-electron many-body basis
#     nbody_basis = []
#     for combination in combinations(range(2 * n_mo), N_electron):
#         fock_state = [0] * (2 * n_mo)
#         for index in list(combination):
#             fock_state[index] += 1
#         nbody_basis += [fock_state]
 
#     return np.array(nbody_basis)  # If pybind11 is used it is better to set dtype=np.int8


# def build_mapping(nbody_basis):
#     """
#     Function to create a unique mapping between a kappa vector and an occupation
#     number state.

#     Parameters
#     ----------
#     nbody_basis :  Many-

#     Returns
#     -------
#     mapping_kappa : List of unique values associated to each kappa
#     """
#     num_digits = np.shape(nbody_basis)[1]
#     dim_H = np.shape(nbody_basis)[0]
#     mapping_kappa = scipy.sparse.lil_matrix((2 ** num_digits, 1) ,dtype=np.int32)
#     for kappa in range(dim_H):
#         ref_state = nbody_basis[kappa]
#         number = 0
#         for digit in range(num_digits):
#             number += ref_state[digit] * 2 ** (num_digits - digit - 1)
#         mapping_kappa[number] = kappa

#     return mapping_kappa

# n_mode  = 2
# N_boson = 1

# nbodybasis = tools_file_bosons.build_nbody_basis(n_mode, N_boson)

# print( nbodybasis )

# mapping = tools_file_bosons.build_mapping( nbodybasis )

# print( mapping ) 

# dim_H = np.shape(nbodybasis)[0]
# kappa = 0

# ref_state = nbodybasis[ kappa ]
# print("refstate", ref_state)

# number = 0
# for index_mode in range(n_mode): 
#     number += ref_state[index_mode] * 10**(n_mode - index_mode - 1)

# print(number)

#%%

# type_of_op = 'a'
# index_mode = 1

# new_state = tools_file_bosons.new_state_after_sq_boson_op(type_of_op, index_mode, ref_state)

# print( "refstate", ref_state )
# print( "newstate", new_state[0], new_state[1] )

n_mode  = 2
n_boson = 8

nbodybasis = tools_file_bosons.build_nbody_basis(n_mode, n_boson)
a_dagger_a = tools_file_bosons.build_operator_a_dagger_a(nbodybasis)

U  = - 0.5 * 100. / n_boson  #PARAM_INI + PARAM_iter * ( PARAM_FIN - PARAM_INI) / N_points
J  = 1.
Mu = 0 #1.e-10 * J
 
h_ = np.zeros(( n_mode, n_mode ))
for site in range(n_mode-1):
    h_[site,site+1] = h_[site+1,site] = -J
h_[0,-1] = h_[-1,0] = -J

U_  = np.zeros(( n_mode, n_mode, n_mode, n_mode )) 
for site in range(n_mode):
    U_[ site, site, site, site ]  = U
    if site==0:
        h_[ site, site ] += - Mu # <=== For Left site 
    else:
        h_[ site, site ] += Mu   # <=== For Right site 

print()
print()

H = tools_file_bosons.build_hamiltonian_bose_hubbard( h_, U_, nbodybasis, a_dagger_a )
eig_en, eig_vec = scipy.linalg.eigh( H.A )
print( eig_en )

#%%


# ref_state = eig_vec[:,0]
# print("refstate", ref_state)

# # state = tools_file_bosons.my_state(ref_state, nbodybasis)

# # print(state)


# tools_file_bosons.visualize_wft(ref_state, nbodybasis)



