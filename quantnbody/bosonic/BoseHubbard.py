from itertools import combinations_with_replacement, product
import numpy as np
from scipy import linalg 
import math as m
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit, prange

N_B = 50
N_S = 2

dim_H = m.factorial(N_B + N_S - 1) // ( m.factorial(N_B)*m.factorial(N_S-1) )
# print("dimension", dim_H)

N_points  = 1
PARAM_FIN = 10.
PARAM_INI = 0.

n_0 = []
n_1 = []
Z = []
sigma_Z = []

NUJ = []

for PARAM_iter in range(N_points):
    
    U  = 4. / N_B   #PARAM_INI + PARAM_iter * ( PARAM_FIN - PARAM_INI) / N_points
    J  = 1.
    Mu = 1.e-10 * J
    
    NUJ += [ N_B*U/J ]
    print()
    print("Value NU/J =", NUJ[-1])
    print()
    Hoppings = np.zeros(( N_S, N_S ))
    for site in range(N_S-1):
        Hoppings[site,site+1] = Hoppings[site+1,site] = -J 
    
    Hoppings[0,-1] = Hoppings[-1,0] = -J
    
    U_site  = np.zeros( N_S )
    Mu_site = np.zeros( N_S )
    for site in range(N_S):
        U_site[ site ]  = U
        if site==0:
            Mu_site[ site ] = - Mu # <=== For Left site 
        else:
            Mu_site[ site ] = Mu   # <=== For Right site 
    
    
    list_focktates = []
    for combination in combinations_with_replacement( range(N_S), N_B ):
        fockstate = [ 0 for i in range(N_S) ]
        for index in list(combination):
            fockstate[ index ] += 1
            
        list_focktates += [ fockstate ]
        # print(fockstate, list(combination))

    H = np.zeros(( dim_H, dim_H ))
    for kappa in tqdm(range(dim_H)):
        ref_fockstate = list_focktates[ kappa ]
        
        for site in range(N_S):
            N_B_site = ref_fockstate[ site ]
            
            if ( N_B_site > 0 ):
                H[ kappa, kappa ] += -0.5 * U_site[ site ] * N_B_site * ( N_B_site - 1. ) - Mu_site[ site ] * N_B_site
                if ( site < N_S ):
                    for new_site in range(site+1,N_S):
                        N_B_new_site  = ref_fockstate[ new_site ]
                        new_fockstate = ref_fockstate.copy()
                        new_fockstate[site]     += -1
                        new_fockstate[new_site] += 1
                        kappa_ = list_focktates.index( new_fockstate )
                        
                        H[ kappa_, kappa ] =  H[ kappa, kappa_ ]  = ( 
                        Hoppings[new_site,site] * m.sqrt( ( N_B_new_site + 1 ) * N_B_site )  )
                        
                        
    eigen_energies, eigen_states = linalg.eigh( H )
    
    fig, ax = plt.subplots()
    ax.matshow( np.abs(eigen_states.T)**2., cmap='viridis' )
    ax.spy( H, marker='.',color='black', markersize=0.1 )
    plt.savefig('matrix_HAM.pdf')  
    
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.step( [ (x)/N_B  for x in range(dim_H)], np.abs( eigen_states[:,0] )**2.,  where="mid", color='black', label='GS' )
    ax.step( [ (x)/N_B  for x in range(dim_H)], np.abs( eigen_states[:,1] )**2.,  where="mid", color='blue', label='GS+1' )
    ax.step( [ (x)/N_B  for x in range(dim_H)], np.abs( eigen_states[:,-1] )**2.,  where="mid", color='red', ls = "--", label='Highest exc. st.' )

    # ax.step( [ (x+1)/dim_H  for x in range(dim_H)], np.abs( eigen_states[:,-2] )**2., color='green', ls = "--" )
    ax.legend()
    ax.set_ylim( 0. )
    ax.set_xlim( 0., 1)
        
    # n_0 += [ N_L/N_B ]
    # n_1 += [ N_R/N_B ]
    # Z   += [ (N_L - N_R)/N_B ]
    # print(Z[-1])


# fig, ax = plt.subplots()
# ax.matshow( np.abs(H), cmap='bone_r' )

# print( H )

# fig, ax = plt.subplots(figsize=(10, 2))
# ax.plot( NUJ, Z, color='black' )
# ax.plot( NUJ, n_0, color='red' )
# ax.plot( NUJ, n_1, color='red' )
# ax.set_ylim( 0., 1 )
# ax.set_xlim( 0., 10)
    


#%%

# from openfermion.ops import BosonOperator
# import openfermion
# import scipy 

# N_B_trunc = N_B  + 1
# Ham = BosonOperator('0^ 0', 0.)
# for site in range(N_S):
#     for site_ in range(site,N_S):
#         Ham += BosonOperator('{}^ {}'.format(site,site_), Hoppings[site,site_])
#         Ham += BosonOperator('{}^ {}'.format(site_,site), Hoppings[site,site_])
        
# for site in range(N_S):
#     Ham += BosonOperator('{0}^ {0} {0}^ {0}'.format(site), -U/2.) - BosonOperator('{0}^ {0}'.format(site), - U/2.)

# print(Ham)

# sparse_HAM = openfermion.boson_operator_sparse(Ham, N_B_trunc)
# # dim = np.shape(sparse_HAM.A)[0]

# # sparse_HAM = sparse_HAM.A[dim-dim_H:,dim-dim_H:]

# number_operator = openfermion.number_operator(N_S, parity=1)  # number_operator(n_modes, mode=None, coefficient=1.0, parity=-1)
# num_sp = openfermion.boson_operator_sparse( number_operator, N_B_trunc ) 

# print("op_num:",number_operator)

# # number_operator_sparse = openfermion.boson_operator_sparse( number_operator, N_B_trunc ) 
# # number_operator_eig = scipy.linalg.eigh(number_operator_sparse.A)
# # NB_projector_basis  = number_operator_eig[1][:, (number_operator_eig[0] == N_B)]
# # NB_projector = np.einsum('ji, ki', NB_projector_basis, NB_projector_basis.conjugate(), optimize=True)  

# # print((NB_projector @ number_operator_sparse @ NB_projector).real)
# # print(np.int_(number_operator_eig[0]))

# # new_ham = NB_projector @ sparse_HAM @ NB_projector
# eig_val, eig_vec = scipy.linalg.eigh( sparse_HAM.A )

# eig_val_filtered = eig_val

# eig_val_filtered = []
# for k in range(len(eig_val)):
    
#     print( int(abs(np.conj(eig_vec[:,k]).T @ num_sp @ eig_vec[:,k])) )
    
#     if( int(abs(np.conj(eig_vec[:,k]) @ num_sp @ eig_vec[:,k])) == N_B   ):
#         eig_val_filtered += [ eig_val[k] ]

# print(  "OF : ", eig_val_filtered)
# print( )
# print( "my code : ",eigen_energies )

# print()
# print((number_operator_eig[0]))
# print(np.int_(number_operator_eig[0]))

# print(eig_vec)

# for val in eigen_energies:
#     print(val)

#%%

