*Tuto 1:* first steps with the package
=====================================================

**Dr. Saad Yalouz - Laboratoire de Chimie Quantique de Strasbourg,
France - July 2022**

Philosophy of the package
-------------------------

The philosophy of the **QuantNBody** package is to facilitate the
implementation and manipulation of quantum many-body systems composed of
electrons or bosons. To achieve this goal, the package has been designed
to provide a fast and easy way to construct many-body operators and wave
functions in a given many-body basis. In this way, it becomes possible
to access quantities/objects of interest in a few lines of python to
speed up and facilitate method development.

To do this, the package works with two fundamental ingredients.

A) The first is the **creation of a many-body basis** (based on a total
   number of quantum particles and modes/orbitals to be filled) in which
   each operator can be represented.

B) The second is the creation of the set of :math:`a^\dagger a`\ \*\*
   hopping operators that are needed to construct any many-body operator
   that conserves the number of particles.

Once these two ingredients are created, the user can use predefined
functions to construct different types of many-body operators
(e.g. Hamiltonians, spin operators), and manipulate/view many-body
quantum states. Note that the QuantNBody package has also been designed
to provide flexibility for users to also create their own operators and
functions based on the tools provided.

**Nota Bene:** For the sake of simplicity, we will concentrate in these
tutorials on fermionic systems.

Let us first import the package !
---------------------------------

.. code:: ipython3

    import quantnbody as qnb   # <==== General import
    import numpy as np

Building a many-body basis
--------------------------

To build a many-body basis for a fermionic system, the QuantNBody
package generates a list of many-body states which describe the
repartition of :math:`N_{elec}` electrons in :math:`2N_{MO}`
spin-orbitals. These states are numerically referenced by a list of
kappa indices such that

.. math::


   \Big\lbrace |\kappa \rangle \Big\rbrace_{\textstyle \kappa=1}^{\textstyle \dim_H}

The dimension :math:`\dim_H` of the many-body basis depends on the
number of electron :math:`N_{elec}` and spatial orbital :math:`N_{MO}`
via a binomial law such that

.. math:: \dim_H = \binom{2N_{MO}}{N_{elec}}

**A little example with** :math:`N_{MO}=N_{elec}=2`: in this case, we
should have **6 many-body states.**

.. code:: ipython3

    N_MO = N_elec = 2 # We define the numebr of MO adn electrons

    nbody_basis = qnb.fermionic.tools.build_nbody_basis( N_MO, N_elec ) # Building the nbody_basis

    print('Shape  of the kappa states')
    for s in range(len(nbody_basis)):
        print('| kappa={} >'.format(s), '=', nbody_basis[s])


.. code:: none

    Shape  of the kappa states
    | kappa=0 > = [1 1 0 0]
    | kappa=1 > = [1 0 1 0]
    | kappa=2 > = [1 0 0 1]
    | kappa=3 > = [0 1 1 0]
    | kappa=4 > = [0 1 0 1]
    | kappa=5 > = [0 0 1 1]


**What is the meaning of these six bit strings ?**

Here, each bit string represents a many-body state. As an example, let
us check the first state for which we have

.. math:: | \kappa  = 0\rangle = | \underbrace{   \overbrace{1}^{ \textstyle  {\alpha}}, \; \; \;\overbrace{1}^{ \textstyle  {\beta}},}_{\textstyle 1st \ MO}\; \; \underbrace{\overbrace{0}^{ \textstyle  {\alpha}}, \; \; \; \overbrace{0}^{ \textstyle  {\beta}}}_{\textstyle 2nd \ MO} \rangle

Here we choose to structure the occupation numbers as follows

-  Each couple of terms refer to **a same spatial orbital**
-  **Even** indices refer to :math:`\alpha`-spinorbitals
-  **Odd** indices refer to :math:`\beta`-spinorbitals

.. note::

  Bluilding a matrix representation of a many-body operator, what does it mean ?
    For each configuration, we associate a unique :math:`\kappa` index which
    defines a unique “numerical” vector. In practice, any numerical
    representation of a given many-body operator will be given in the numerical many-body basis
    indexed by the :math:`\kappa`. As an example, let us imagine we want to
    encode numerically a second quantization operator :math:`O`. This means
    in practice that we create a matrix representation of this operator in the many-body
    basis such that

    .. math::

        O = \sum_{\kappa, \kappa'
        =1}^{\dim_H}  \langle \kappa' | O | \kappa  \rangle  \; | \kappa'    \rangle\langle \kappa |

    In practice, this indexing is realized by the QuantNBody package and
    used then as a central tool to build every matrix element of a given
    many-body operators.

Building and storing the :math:`a^\dagger_{p,\sigma} a_{q,\tau}` operators
--------------------------------------------------------------------------

Once the list of many-body state is created, the next crucial point in
the **QuantNBody** package consists in building the
:math:`a^\dagger_{p,\sigma} a_{q,\tau}` many-body operators.

In practice, these operators play a central role in many cases of study
as soon as we have to deal with **systems that are particle-number
conserving.** In this case, one can show that many objects
(i.e. excitation operators, spin operators, reduced density matrices …)
are built in practice using series of
:math:`a^\dagger_{p,\sigma} a_{q,\tau}` operators.

With the QuantNBody package, we build the
:math:`a^\dagger_{p,\sigma} a_{q,\tau}` operators once and for all and
store them via a very simple command line. This way we will be able to
use them later on for any type of developments.

The command line is simple and only require the list of many-body states
we built previously :

.. code:: ipython3

    a_dagger_a = qnb.fermionic.tools.build_operator_a_dagger_a( nbody_basis )

**How to get access to these operators once stored ?**

The way each operator is stored follows the way we order the
spin-orbitals in our many-body states. As an illustrative example,
taking the following elements will return the associated many-body
operators :

.. raw:: html

   <center>

a_dagger_a[0,0] :math:`\longrightarrow a^\dagger_{0,\alpha} a_{0,\alpha}`

.. raw:: html

   </center>

.. raw:: html

   <center>

a_dagger_a[1,0] :math:`\longrightarrow a^\dagger_{0,\beta} a_{0,\alpha}`

.. raw:: html

   </center>

.. raw:: html

   <center>

a_dagger_a[10,1]  :math:`\longrightarrow a^\dagger_{5,\alpha} a_{0,\beta}`

.. raw:: html

   </center>

In practice, the resulting many-body operators we get access to are
expressed in the original many-body basis stored under a sparse format.
We take the example of the first operator :math:`a^\dagger_{0,\alpha}a_{0,\alpha}` below for which we show the asscociated
sparse and dense matrix representation in the many-body basis

.. code:: ipython3

    print(  "Sparse representation of a_dagger_a[0,0]" )
    print( a_dagger_a[0,0] )

    print( )
    print( "Dense representation of a_dagger_a[0,0]" )
    print( a_dagger_a[0,0].A )


.. code:: none

    Sparse representation of a_dagger_a[0,0]
      (0, 0)	1.0
      (1, 1)	1.0
      (2, 2)	1.0

    Dense representation of a_dagger_a[0,0]
    [[1. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]]


We see here that this operator simply counts the number of electrons in
the first spin-orbital explaining why we only have ones on the three
first elements of the diagonal (see the shape of the three many-body
states given ealrier to understand).

Building our first many-body Hamiltonian : a fermi-Hubbard molecule
-------------------------------------------------------------------

In this final part of the tutorial we will use the previously built
:code:a_dagger_a variable to implement a fermi-Hubbard molecule. In the local
site basis, the model Hamiltonian is usually expressed such that:

.. math::


   \hat{H} = \color{blue}{\sum_{\langle i,j \rangle}^{N_{MO}} -t_{ij} \sum_{\sigma=\uparrow,\downarrow} (\hat{a}^\dagger_{j,\sigma}\hat{a}_{i,\sigma}+\hat{a}^\dagger_{i,\sigma}\hat{a}_{j,\sigma})}
   + \color{red}{\sum_i^{N_{MO}} \mu_{ii} \sum_{\sigma=\uparrow,\downarrow} \hat{a}^\dagger_{i,\sigma}\hat{a}_{i,\sigma} }
   + \color{black}{
   \sum_i^{N_{MO}} U_{iiii} \hat{a}^\dagger_{i,\uparrow}\hat{a}_{i,\uparrow} \hat{a}^\dagger_{i,\downarrow}\hat{a}_{i,\downarrow}
   }

| with :
- :math:`t_{ij}` the hopping terms between the pair of
  connected sites :math:`\langle i, j \rangle`.

- :math:`\mu_{ii}` the local chemical potential on site “:math:`i`”.

- :math:`U_{iiii}` the local coulombic repulsion on site “:math:`i`”.


We illustrate the shape of the system below
   .. image:: graph.png
      :width: 300
      :align: center

In a more general basis (not necessarily local) we have

.. math::


   \hat{H} =\sum_{\langle p,q \rangle}^{N_{MO}} -h_{pq} \sum_{\sigma=\uparrow,\downarrow} (\hat{a}^\dagger_{p,\sigma}\hat{a}_{q,\sigma}+\hat{a}^\dagger_{q,\sigma}\hat{a}_{p,\sigma}) + \sum_i^{N_{MO}} U_{p,q,r,s} \hat{a}^\dagger_{p,\uparrow}\hat{a}_{q,\uparrow} \hat{a}^\dagger_{r,\downarrow}\hat{a}_{s,\downarrow} ,


where for commodity we have introduced the one-body integrals
:math:`h_{pq}` which embed the hopping terms and the chemical potentials
such as

.. math::


   h_{pq} = \sum_{i,j}^{N_{MO}} (-t_{ij} + \delta_{ij}\mu_{ii}) C_{i,p} C_{j,q},

and the “delocalized version” of the coulombic repulsion term

.. math::


   U_{pqrs} = \sum_{i}^{N_{MO}}  U_{i,i,i,i} C_{i,p} C_{i,q} C_{i,r} C_{i,s},

where the matrix :math:`{\bf C}` encodes the Molecular Orbital
coefficients (used if we want for example to express the Hamiltonian in
a delocalized basis).

**Building the Hamiltonian :** To initiate the construction of the
matrix representation of the operator in the many-body basis, we first
define the hopping term :math:`t` between the sites, the chemical
potentials :math:`\mu` and the electronic repulsion :math:`U`.

.. code:: ipython3

    # Setup for the simulation ========
    N_MO   = N_elec = 2
    t_  = np.zeros((N_MO,N_MO))
    U_  = np.zeros((N_MO,N_MO,N_MO,N_MO))
    Mu_ = np.zeros((N_MO,N_MO))
    for i in range(N_MO):
        U_[i,i,i,i]  =  1 * (1+i)  # Local coulombic repulsion
        Mu_[i,i]     = -1 * (1+i)  # Local chemical potential

        for j in range(i+1,N_MO):
            t_[i,j] = t_[j,i] = - 1  # hopping

    h_ = t_  + np.diag( np.diag(Mu_) ) # Global one-body matrix = hoppings + chemical potentials

    print( 't_=\n',t_ ,'\n')

    print( 'Mu_=\n',Mu_ ,'\n')

    print( 'h_=\n',h_ ,'\n')


.. code:: none

    t_=
     [[ 0. -1.]
     [-1.  0.]]

    Mu_=
     [[-1.  0.]
     [ 0. -2.]]

    h_=
     [[-1. -1.]
     [-1. -2.]]



To build the Hamiltonian, we simply have to pass the three following
ingredients to an already built function:

- Parameters of the model
- The Many-body basis
- The :math:`a^\dagger a` operators

as shown below

.. code:: ipython3

    H_fermi_hubbard = qnb.fermionic.tools.build_hamiltonian_fermi_hubbard( h_,
                                                                           U_,
                                                                           nbody_basis,
                                                                           a_dagger_a )

Similarily to the :math:`a^\dagger a` operators, the Hamiltonian :math:`H` is
represented in the many-body basis with a native sparse representation
(which can be made dense):

.. code:: ipython3

    print('H (SPARSE) =' )
    print(H_fermi_hubbard)

    print()
    print('H (DENSE) =' )
    print(H_fermi_hubbard.A)


.. code:: none

    H (SPARSE) =
      (0, 0)	-1.0
      (0, 2)	-1.0
      (0, 3)	1.0
      (1, 1)	-3.0
      (2, 0)	-1.0
      (2, 2)	-3.0
      (2, 5)	-1.0
      (3, 0)	1.0
      (3, 3)	-3.0
      (3, 5)	1.0
      (4, 4)	-3.0
      (5, 2)	-1.0
      (5, 3)	1.0
      (5, 5)	-2.0

    H (DENSE) =
    [[-1.  0. -1.  1.  0.  0.]
     [ 0. -3.  0.  0.  0.  0.]
     [-1.  0. -3.  0.  0. -1.]
     [ 1.  0.  0. -3.  0.  1.]
     [ 0.  0.  0.  0. -3.  0.]
     [ 0.  0. -1.  1.  0. -2.]]


Once :math:`H` is built, we can diagonalize the resulting matrix using
for example the numpy library.

.. code:: ipython3

    eig_energies, eig_vectors =  np.linalg.eigh(H_fermi_hubbard.A)

    print('Energies =', eig_energies[:4] )


.. code:: none

    Energies = [-4.41421356 -3.         -3.         -3.        ]


And finally, we can call a very useful function from the QuantNBody
package that help visualizing the shape of a wavefunction as shown
below. This function lists the most important many-body state
contributing to the wavefunction with the associated coefficients in
front.

.. code:: ipython3

    WFT_to_analyse = eig_vectors[:,0]

    # Visualizing the groundstate in the many-body basis
    qnb.fermionic.tools.visualize_wft( WFT_to_analyse, nbody_basis ) # <=== FCT IN THE PACKAGE
    print()


.. code:: none


    	-----------
    	 Coeff.      N-body state
    	-------     -------------
    	-0.57454	|0110⟩
    	+0.57454	|1001⟩
    	+0.47596	|0011⟩
    	+0.33656	|1100⟩
