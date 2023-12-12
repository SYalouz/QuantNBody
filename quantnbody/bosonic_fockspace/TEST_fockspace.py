from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_general # bases 
import numpy as np # general math functions 
import scipy 
import quantnbody as qnb

##### setting up parameters for simulation
L       = 3
N_min   = 0
N_max   = 5

List_J  = [ np.random.rand() for i in range(L-1) ]
List_U  = [ np.random.rand() for i in range(L) ] 

List_Nb = [ i for i in range(N_min, N_max+1) ]   
            
###### create the basis
# build the two bases to tensor together to a bose-fermi mixture 
boson_basis = boson_basis_general( L, Nb=List_Nb ) # fermion basis  

##### create model  
hop_f_right  = [ [ List_J[i],i,i+1 ] for i in range(L-1) ]  # f hopping right
hop_f_left   = [ [ List_J[i],i,i+1 ] for i in range(L-1) ]  # f hopping left
int_ff       = [ [ List_U[i],i,i ] for i in range(L) ]      # ff nearest-neighbour interaction  
int_ff_      = [ [ -List_U[i],i ] for i in range(L) ]      # ff nearest-neighbour interaction  
  
# create static lists
static = [	["+-", hop_f_left],  # fermions hop left
			["-+", hop_f_right], # fermions hop right
			["nn", int_ff], # fermionfermion on-site interactions  
            ["n", int_ff_], # fermionfermion on-site interactions
			]

dynamic = []  

###### set up Hamiltonian and initial states
no_checks = dict(check_pcon=False,check_symm=False,check_herm=True)

H = hamiltonian(static,dynamic,basis=boson_basis,**no_checks)

# compute GS of H 
E_, psi_=  scipy.linalg.eigh( H.todense() )

dim_H = np.shape(H.todense())[0]

# print( E_[0:10] ) 

#%%

List_Nb = [ i for i in range(N_min, N_max+1) ] 
nbody_basis = qnb.bosonic_fockspace.tools.build_nbody_basis( L,
                                                             List_Nb )

dim_H_ = len(nbody_basis)

a = qnb.bosonic_fockspace.tools.build_anihilation_operator_a( nbody_basis )
 
Num_op = 0
a_dagger_a = np.zeros((L,L), dtype=object) 
for p in range( L ): 
    Num_op += a[p].T @ a[p]
    for q in range( L ): 
        a_dagger_a[p,q] =   a[p].T @ a[q]  
# print("OK") 
 
h_ = np.zeros((L,L))
U_ = np.zeros((L,L,L,L))
for i in range(L-1):
    h_[i,i+1]  = h_[i+1,i] = List_J[i]
for i in range(L): 
    U_[i,i,i,i] = List_U[i]
    
H_ = qnb.bosonic_fockspace.tools.build_hamiltonian_bose_hubbard(h_, U_, nbody_basis, a_dagger_a)

# H = H + 1000 * (Num_op - n_mo * scipy.sparse.identity(np.shape(H)[0]))**2
# H = H +  S_Z@S_Z  #+ S_2 
# eig_val, eig_vec = scipy.sparse.linalg.eigsh(H, which='SA',k=20)
eig_val, eig_vec = scipy.linalg.eigh(H_.A)
# print(eig_val[0:10]) 

print( np.allclose(H_.A ,H_.A.T) )

print( E_[0:10] )
print()
print( eig_val[0:10] )
print()
print( np.allclose(E_ ,eig_val) )
print('dims:', dim_H, dim_H_ )

