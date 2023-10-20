*Tuto 4:* the Bose-Hubbard system
=================================

**Dr. Saad Yalouz - Laboratoire de Chimie Quantique de Strasbourg,
France - July 2022**

Introduction : the Bose-Hubbard system
--------------------------------------

The Bose-Hubbard Hamiltonian is defined in the site basis as follows

.. math::  H = \sum_{i,j} h_{ij} a^\dagger_i a_j + \sum_{i} U_{iiii}  a^\dagger_i a^\dagger_i a_i a_i ,

where :math:`h_{ij}` and :math:`U_{iiii}` are the two- and four-indices integrals
associated respectively to the single-body and the two-body part of the
full Hamiltonian.

In an extended basis (i.e. in a basis of non-local modes), the same
Hamiltonian takes the most general form

.. math::  H = \sum_{p,q} h_{pq} a^\dagger_p a_q + \sum_{p,q,r,s} U_{pqrs}  a^\dagger_p a^\dagger_q a_r a_s,

where the single-body integrals have been transformed (thanks to a
transfer matrix :math:`C`) into an extended basis such that

.. math::  h_{pq} = \sum_{i,j} C_{ip} C_{jq} h_{ij},

and similarily for the two-body integrals

.. math::  U_{pqrs} = \sum_{i,j} C_{ip} C_{iq} C_{ir} C_{is} U_{iiii}.

In this more general context, the energy of a given reference many-body
state :math:`| \Psi \rangle` is defined as follows

.. math::  E_{ |\Psi \rangle} = \sum_{p,q} h_{pq} \gamma_{pq} + \sum_{p,q,r,s} U_{pqrs} \Gamma_{pqrs},

where :math:`\gamma` represents the so-called 1-RDM whose elements are
defined like

.. math::  \gamma_{pq} = \langle \Psi | a^\dagger_p a_q  | \Psi \rangle ,

and :math:`\Gamma` the 2-RDM with the elements

.. math::  \Gamma_{pqrs} = \langle \Psi | a^\dagger_p a^\dagger_q a_r a_s | \Psi \rangle

We show below how to build all these objects with QuantNBody.

Importing the required libraries
--------------------------------

.. code:: ipython3

    import quantnbody as qnb
    import scipy
    import numpy as np

Defining the basic properties of the system
-------------------------------------------

.. code:: ipython3

    n_mode  = 5 # Number of modes
    n_boson = 5 # Number of bosons

    # Building the one-body tensor in a general extended basis
    h_tensor = np.zeros(( n_mode, n_mode ))
    for site in range(n_mode):
        for site_ in range(n_mode):
            if (site != site_):
                h_tensor[site,site_] = h_tensor[site_,site] = -1 # <== a lattice fully connected with a same hopping term

    # Building the two-body tensor in a general extended basis
    U_tensor  = np.zeros(( n_mode, n_mode, n_mode, n_mode ))
    for site in range(n_mode):
        U_tensor[ site, site, site, site ]  = - 10.1 # <=== Local on-site attraction of the bosons

    # # Uncomment below in case we want to switch to an extended basis
    # eig_h, C = scipy.linalg.eigh( h_tensor ) # <== Extended basis simply given by the eigenmode of h_tensor
    # h_tensor, U_tensor = qnb.bosonic.tools.transform_1_2_body_tensors_in_new_basis(h_tensor, U_tensor, C)

Building the essential tools for the QuantNBody package
-------------------------------------------------------

.. code:: ipython3

    # Building the many-body basis
    nbodybasis = qnb.bosonic.tools.build_nbody_basis( n_mode, n_boson )

    # Building the a†a operators
    a_dagger_a = qnb.bosonic.tools.build_operator_a_dagger_a( nbodybasis )

All-in-one function
-------------------

We define below an “all-in-one” function that returns :

- Bose-Hubbard Hamiltonian
- Groundstate FCI energy
- Groundstate wavefunction
- Groundstate 1- and 2-RDMs.

.. code:: ipython3

    def Bose_hubbard_all_in_one( h_tensor, U_tensor, nbodybasis, a_dagger_a ):

        # Building the matrix representation of the Hamiltonian operators
        Hamiltonian = qnb.bosonic.tools.build_hamiltonian_bose_hubbard( h_tensor,
                                                                        U_tensor,
                                                                        nbodybasis,
                                                                        a_dagger_a )
        eig_en, eig_vec = scipy.linalg.eigh( Hamiltonian.A  )

        GS_WFT     = eig_vec[:,0]
        GS_energy  = eig_en[0]
        GS_one_rdm = qnb.bosonic.tools.build_1rdm( GS_WFT, a_dagger_a )
        GS_two_rdm = qnb.bosonic.tools.build_2rdm( GS_WFT, a_dagger_a )

        return Hamiltonian, GS_energy, GS_WFT, GS_one_rdm, GS_two_rdm

Applying the function to get information from the system
--------------------------------------------------------

.. code:: ipython3

    Hamiltonian, GS_energy, GS_WFT, GS_one_rdm, GS_two_rdm = Bose_hubbard_all_in_one( h_tensor,
                                                                                      U_tensor,
                                                                                      nbodybasis,
                                                                                      a_dagger_a )

Visualizing the resulting wavefunction in the many-body basis
-------------------------------------------------------------

.. code:: ipython3

    qnb.bosonic.tools.visualize_wft( GS_WFT, nbodybasis )
    print()


.. code:: none


    	-----------
    	 Coeff.      N-body state
    	-------     -------------
    	+0.44648	|0,0,5,0,0⟩
    	+0.44648	|0,0,0,0,5⟩
    	+0.44648	|0,0,0,5,0⟩
    	+0.44648	|0,5,0,0,0⟩
    	+0.44648	|5,0,0,0,0⟩
    	+0.01283	|0,0,4,0,1⟩
    	+0.01283	|0,0,4,1,0⟩
    	+0.01283	|0,1,4,0,0⟩




Checking the implementation : comparing different ways to estimate the groundstate energy
-----------------------------------------------------------------------------------------

In order to check if everything is correct, we can compare the resulting
GS energy. First, let us evaluate it via the left/right projections on the Hamiltonian :math:`\langle  \Psi | H |\Psi\rangle` as shown below

.. code:: ipython3

    E_projection = GS_WFT.T @ Hamiltonian @ GS_WFT # <== Very simple and intuitive

Then using our knowledge of the groundstate RDMs (as shown at the
begining of the notebook), this can be done like this

.. code:: ipython3

    E_with_RDMs = ( np.einsum( 'pq,pq->', h_tensor, GS_one_rdm, optimize=True)        # <== A bit more elaborated
                +   np.einsum( 'pqrs,pqrs->', U_tensor, GS_two_rdm, optimize=True)  )

And we can finally compare all these results to the one provided by the
“all in one function” :

.. code:: ipython3

    print("GS energy estimations ======================== ")
    print( "With the all in one function", E_projection )
    print( "With the projection method  ", E_projection )
    print( "With the RDMs method        ", E_with_RDMs )


.. code:: none

    GS energy estimations ========================
    With the all in one function -202.25704161029097
    With the projection method   -202.25704161029097
    With the RDMs method         -202.257041610291


we should obtain exactly the same thing !
