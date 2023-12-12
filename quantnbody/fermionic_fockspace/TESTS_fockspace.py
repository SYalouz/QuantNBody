import quantnbody as qnb
import numpy as np
import scipy

List_J  = [ 1, 1.5 ]
List_U  = [ 1, 2, 3  ] 
n_mo    = 3
list_N_electron = [ i for i in range(5) ]
# list_N_electron = [  0, 1, 2, 3  ]

nbody_basis = qnb.fermionic_fockspace.tools.build_nbody_basis( n_mo,
                                                               list_N_electron )
 
dim_H = len(nbody_basis)

for ind in range(dim_H):
    print(nbody_basis[ind], ' ', ind)

a = qnb.fermionic_fockspace.tools.build_anihilation_operator_a( nbody_basis )

# print()
# for p in range(2*n_mo):
#     print(p)
#     print(a[p])
#     print()
 

Num_op = 0
a_dagger_a = np.zeros((2*n_mo,2*n_mo), dtype=object) 
for p in range( 2*n_mo ): 
    Num_op += a[p].T @ a[p]
    for q in range( 2*n_mo ): 
        a_dagger_a[p,q] =   a[p].T @ a[q]  
        
        # H = a_dagger_a[p,q] + a_dagger_a[p,q].T
        
# print("OK")

# S_2, S_Z, S_plus = qnb.fermionic_fockspace.tools.build_s2_sz_splus_operator(a_dagger_a)

# %% 

h_ = np.zeros((n_mo,n_mo))
U_ = np.zeros((n_mo,n_mo,n_mo,n_mo))
for i in range(n_mo-1):
    h_[i,i+1]  = h_[i+1,i] = List_J[i]
for i in range(n_mo): 
    U_[i,i,i,i] = List_U[i]
    
H = qnb.fermionic_fockspace.tools.build_hamiltonian_fermi_hubbard(h_, U_, nbody_basis, a_dagger_a)

# H = H + 1000 * (Num_op - n_mo * scipy.sparse.identity(np.shape(H)[0]))**2
# H = H +  S_Z@S_Z  #+ S_2 
# eig_val, eig_vec = scipy.sparse.linalg.eigsh(H, which='SA',k=20)
eig_val, eig_vec = scipy.linalg.eigh(H.A)
print(eig_val[0:10])