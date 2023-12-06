import quantnbody as qnb
import numpy as np
import scipy

U = 20
t = -1
n_mo = 2
# list_N_electron = [ i for i in range(1,3) ]
list_N_electron = [ 0, 1, 2, 3 , 4 ]

nbody_basis = qnb.fermionic_fockspace.tools.build_nbody_basis( n_mo, list_N_electron )

dim_H = len(nbody_basis)

a = qnb.fermionic_fockspace.tools.build_anihilation_operator_a( nbody_basis )

print()
for p in range(2*n_mo):
    print(p)
    print(a[p])
    print()

# Num_op = 0
# a_dagger_a = np.zeros((2*n_mo,2*n_mo), dtype=object) 
# for p in range( 2*n_mo ): 
#     Num_op += a[p].T @ a[p]
#     for q in range( 2*n_mo ): 
#         a_dagger_a[p,q] =   a[p].T @ a[q]  
# print("OK")

#%%


# print(np.allclose(H.A, np.conj(H.A).T))

# h_ = np.zeros((n_mo,n_mo))
# U_ = np.zeros((n_mo,n_mo,n_mo,n_mo))
# for i in range(n_mo-1):
#     h_[i,i+1]  = h_[i+1,i] = t
# for i in range(n_mo): 
#     U_[i,i,i,i] =   U  
    
# H = qnb.fermionic_fockspace.tools.build_hamiltonian_fermi_hubbard(h_, U_, nbody_basis, a_dagger_a)

# H = H + 1000 * (Num_op - n_mo * scipy.sparse.identity(np.shape(H)[0]))**2

# eig_val, eig_vec = scipy.sparse.linalg.eigsh(H, which='SA')

# print(eig_val)