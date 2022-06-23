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

n_mode  = 4
n_boson = 20

nbodybasis = tools_file_bosons.build_nbody_basis(n_mode, n_boson)

# print( nbodybasis[115], nbodybasis[120] )

mapping = tools_file_bosons.build_mapping( nbodybasis )

print( mapping ) 

dim_H = np.shape(nbodybasis)[0]
kappa = 0

ref_state = nbodybasis[ kappa ]
print("refstate", ref_state)

number = 0
for index_mode in range(n_mode): 
    number += ref_state[index_mode] * 10**(n_mode - index_mode - 1)

print(number)

 
nbodybasis = tools_file_bosons.build_nbody_basis(n_mode, n_boson)
a_dagger_a = tools_file_bosons.build_operator_a_dagger_a(nbodybasis)

U  = - 0.5 * 10. / n_boson  #PARAM_INI + PARAM_iter * ( PARAM_FIN - PARAM_INI) / N_points
J  = 1.
Mu = 0. #1.e-10 * J

h_ = np.zeros(( n_mode, n_mode ))
for site in range(n_mode-1):
    h_[site,site+1] = h_[site+1,site] = -J
h_[0,-1] = h_[-1,0] = -J

U_  = np.zeros(( n_mode, n_mode, n_mode, n_mode )) 
for site in range(n_mode):
    U_[ site, site, site, site ]  = U

print() 

H = tools_file_bosons.build_hamiltonian_bose_hubbard( h_, U_, nbodybasis, a_dagger_a )
eig_en, eig_vec = scipy.linalg.eigh( H.A ) 
print( eig_en[:20] )

#%%

# H_full = H.A

ref_state =  eig_vec[:,0]
# print("refstate", ref_state)

# state = tools_file_bosons.my_state(ref_state, nbodybasis)

# print(state)
# 

# tools_file_bosons.visualize_wft( ref_state, nbodybasis )

# # print()
# # print(H.A)

# print(np.max(H_full))
# print(np.min(H_full))

# print(np.where( H_full == np.min(H_full) ))

# print(np.allclose(H_full, H_full.T))
# op = a_dagger_a + a_dagger_a.T
# print(np.allclose(op, op.T))

#%%

# ref_state = nbodybasis[120]

# mapping_kappa = tools_file_bosons.build_mapping( nbodybasis )
# result = tools_file_bosons.build_final_state_ad_a(ref_state, 1, 1, mapping_kappa)
# print(result)

# p = 1

# print(  - 0.5 * 100. / n_boson * 
#       state.T @ a_dagger_a[ p , p ] @ ( a_dagger_a[ p , p ] - scipy.sparse.identity(dim_H) ) @  state )

# ref_state = np.zeros((dim_H)) #eig_vec[:,0] # 
# ref_state[0] = 1
# tools_file_bosons.visualize_wft( ref_state, nbodybasis )
# print( tools_file_bosons.make_number_out_of_vector(ref_state) )



# ref_state = [ 2, 2, 11]
# print( ref_state )
# print( len(ref_state) ) 
# print(  np.sum(ref_state) )
# print(  len(str(n_boson)) )

# number = 0
# for index_mode in range(n_mode):
#     number += ref_state[index_mode] * 10**(( n_mode - index_mode - 1 )*2)
# print(number)


