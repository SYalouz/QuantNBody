*QuantNbody* tutorials : playing with many-body wavefunctions
=============================================================

**Dr. Saad Yalouz - Laboratoire de Chimie Quantique de Strasbourg,
France - July 2022**

In this second QuantNBody tutorial we will focus on the manipulation of
states with different illustrative examples. For this, we will consider
a system composed of :math:`N_e=4` electrons in :math:`N_{MO} = 4`
molecular orbitals (so 8 spinorbitals in total).

We first import the package and then define these properties

.. code:: ipython3

    import quantnbody as qnb
    import numpy as np
    import scipy

    N_MO = N_elec = 4 # Number of MOs and electrons in the system

Building first the many-body basis and the :math:`a^\dagger_{p,\sigma} a_{q,\tau}` operators
--------------------------------------------------------------------------------------------

.. code:: ipython3

    nbody_basis = qnb.fermionic.tools.build_nbody_basis( N_MO, N_elec )
    a_dagger_a = qnb.fermionic.tools.build_operator_a_dagger_a( nbody_basis )

    print('The many-body basis')
    print(nbody_basis)


.. parsed-literal::

    The many-body basis
    [[1 1 1 1 0 0 0 0]
     [1 1 1 0 1 0 0 0]
     [1 1 1 0 0 1 0 0]
     [1 1 1 0 0 0 1 0]
     [1 1 1 0 0 0 0 1]
     [1 1 0 1 1 0 0 0]
     [1 1 0 1 0 1 0 0]
     [1 1 0 1 0 0 1 0]
     [1 1 0 1 0 0 0 1]
     [1 1 0 0 1 1 0 0]
     [1 1 0 0 1 0 1 0]
     [1 1 0 0 1 0 0 1]
     [1 1 0 0 0 1 1 0]
     [1 1 0 0 0 1 0 1]
     [1 1 0 0 0 0 1 1]
     [1 0 1 1 1 0 0 0]
     [1 0 1 1 0 1 0 0]
     [1 0 1 1 0 0 1 0]
     [1 0 1 1 0 0 0 1]
     [1 0 1 0 1 1 0 0]
     [1 0 1 0 1 0 1 0]
     [1 0 1 0 1 0 0 1]
     [1 0 1 0 0 1 1 0]
     [1 0 1 0 0 1 0 1]
     [1 0 1 0 0 0 1 1]
     [1 0 0 1 1 1 0 0]
     [1 0 0 1 1 0 1 0]
     [1 0 0 1 1 0 0 1]
     [1 0 0 1 0 1 1 0]
     [1 0 0 1 0 1 0 1]
     [1 0 0 1 0 0 1 1]
     [1 0 0 0 1 1 1 0]
     [1 0 0 0 1 1 0 1]
     [1 0 0 0 1 0 1 1]
     [1 0 0 0 0 1 1 1]
     [0 1 1 1 1 0 0 0]
     [0 1 1 1 0 1 0 0]
     [0 1 1 1 0 0 1 0]
     [0 1 1 1 0 0 0 1]
     [0 1 1 0 1 1 0 0]
     [0 1 1 0 1 0 1 0]
     [0 1 1 0 1 0 0 1]
     [0 1 1 0 0 1 1 0]
     [0 1 1 0 0 1 0 1]
     [0 1 1 0 0 0 1 1]
     [0 1 0 1 1 1 0 0]
     [0 1 0 1 1 0 1 0]
     [0 1 0 1 1 0 0 1]
     [0 1 0 1 0 1 1 0]
     [0 1 0 1 0 1 0 1]
     [0 1 0 1 0 0 1 1]
     [0 1 0 0 1 1 1 0]
     [0 1 0 0 1 1 0 1]
     [0 1 0 0 1 0 1 1]
     [0 1 0 0 0 1 1 1]
     [0 0 1 1 1 1 0 0]
     [0 0 1 1 1 0 1 0]
     [0 0 1 1 1 0 0 1]
     [0 0 1 1 0 1 1 0]
     [0 0 1 1 0 1 0 1]
     [0 0 1 1 0 0 1 1]
     [0 0 1 0 1 1 1 0]
     [0 0 1 0 1 1 0 1]
     [0 0 1 0 1 0 1 1]
     [0 0 1 0 0 1 1 1]
     [0 0 0 1 1 1 1 0]
     [0 0 0 1 1 1 0 1]
     [0 0 0 1 1 0 1 1]
     [0 0 0 1 0 1 1 1]
     [0 0 0 0 1 1 1 1]]


Building our own many-body wavefunction
---------------------------------------

The package QuantNBody offers the possibility to define our very own
many-body wavefunction in an intuitive manner. For this we can use the
function “my_state” to transform any occupation number state
(handwritten in the code) into a referenced state in the numerical
representation of the many-body basis (i.e. the :math:`| \kappa \rangle`
states).

As a demonstration, let us imagine that we want to build a simple slater
determinant

.. math:: | \Psi \rangle = |00001111\rangle

we show below how do that

.. code:: ipython3

    State_to_translate = [ 0,0,0,0,1,1,1,1]

    Psi = qnb.fermionic.tools.my_state( State_to_translate, nbody_basis )

    print( Psi )


.. parsed-literal::

    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]


As shown here, printing the state returns a vector of dimension equal to
the number of configurations. The last state of the many-body basis is
indeed the one we want to encode explaining why we have a coefficient 1
in the last position. This is normal as here we translate an occupation
number vector to its respective many-body :math:`\kappa` state encoded
numerically (see the first tutorial).

Naturally, we can go beyond the previous simple example and try to
create a multi-configurational wavefunction. As an example, let us
consider the following wavefunction to be encoded numerically

.. math:: | \Psi \rangle = (|00001111\rangle + |11110000\rangle)/\sqrt{2}.

We show below how to do that

.. code:: ipython3

    State_to_translate = [ 0,0,0,0,1,1,1,1]

    Psi = qnb.fermionic.tools.my_state( State_to_translate, nbody_basis )

    State_to_translate = [1,1,1,1,0,0,0,0]
    Psi += qnb.fermionic.tools.my_state( State_to_translate, nbody_basis )

    Psi = Psi/np.sqrt(2)

    print( Psi )


.. parsed-literal::

    [0.70710678 0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.70710678]


In this second case, we obtain a :math:`1/\sqrt{2}` factor on the first
and last positions of the vector which is expected. As a simple check of
our implementation, we can also visualize the final wavefunction we have
just built using the “visualize_wft” function implemented in QuantNBody:

.. code:: ipython3

    qnb.fermionic.tools.visualize_wft( Psi, nbody_basis )
    print()


.. parsed-literal::


    	-----------
    	 Coeff.      N-body state
    	-------     -------------
    	+0.70711	|00001111⟩
    	+0.70711	|11110000⟩




Which returns precisely what we have implemented !

Building filtered lists of many-body states
-------------------------------------------

A particularily interesting action we can realize is to filter the
many-body basis to only retain states that respect a particular
property. As an example, let us imagine that we want to create a list of
neutral states with only one electron by molecular orbital at most. We
show below one possible way to filter the many-body basis using the
a_dagger_a variable.

.. code:: ipython3

    dim_total = len(nbody_basis)

    Op_filtering = ( a_dagger_a[0, 0] + a_dagger_a[1, 1]  -  scipy.sparse.identity(dim_total) )**2
    for p in range(1,N_MO):
        Op_filtering  +=   (a_dagger_a[2*p, 2*p] + a_dagger_a[2*p+1, 2*p+1] -  scipy.sparse.identity(dim_total) )**2

    list_index_det_neutral  = np.where( (np.diag( Op_filtering.A ) == 0.)  )[0]


    print()
    print(" List of neutral states obtained ")
    for index in list_index_det_neutral:
        print(nbody_basis[index])


.. parsed-literal::


     List of neutral states obtained
    [1 0 1 0 1 0 1 0]
    [1 0 1 0 1 0 0 1]
    [1 0 1 0 0 1 1 0]
    [1 0 1 0 0 1 0 1]
    [1 0 0 1 1 0 1 0]
    [1 0 0 1 1 0 0 1]
    [1 0 0 1 0 1 1 0]
    [1 0 0 1 0 1 0 1]
    [0 1 1 0 1 0 1 0]
    [0 1 1 0 1 0 0 1]
    [0 1 1 0 0 1 1 0]
    [0 1 1 0 0 1 0 1]
    [0 1 0 1 1 0 1 0]
    [0 1 0 1 1 0 0 1]
    [0 1 0 1 0 1 1 0]
    [0 1 0 1 0 1 0 1]


Similarily we can also search only the doubly occupied state
(i.e. seniority zero configurations) which could be done via a small
modification of what has been proposed before

.. code:: ipython3

    Op_filtering = ( a_dagger_a[0, 0] + a_dagger_a[1, 1]  -  2*scipy.sparse.identity(dim_total) )**2
    for p in range(1,N_MO):
        Op_filtering  +=   (a_dagger_a[2*p, 2*p] + a_dagger_a[2*p+1, 2*p+1] -  2* scipy.sparse.identity(dim_total) )**2

    list_index_det_neutral  = np.where( (np.diag( Op_filtering.A ) == 8)  )[0]


    print()
    print(" List of doubly occupied states obtained ")
    for index in list_index_det_neutral:
        print(nbody_basis[index])



.. parsed-literal::


     List of doubly occupied states obtained
    [1 1 1 1 0 0 0 0]
    [1 1 0 0 1 1 0 0]
    [1 1 0 0 0 0 1 1]
    [0 0 1 1 1 1 0 0]
    [0 0 1 1 0 0 1 1]
    [0 0 0 0 1 1 1 1]


Applying excitations to a state
-------------------------------

In this final part we show the effect of applying excitations to a
reference wavefunction. For this, we will consider implementing a
singlet excitation over an initial configuration to produce the final
state

.. math::  | \Psi \rangle = (a^\dagger_{3,\alpha}a_{2,\alpha} + a^\dagger_{3,\beta}a_{2,\beta})| 11110000\rangle / \sqrt{2}

This is very easy to implement with the QuantNBody package. In this
case, as shown below, the second quantization algebra can be very
straightforwardly implemented in a few line of python code !

.. code:: ipython3

    # We first translate the occupation number config into the many-body basis of kappa vectors
    initial_config_occ_number = [ 1, 1, 1, 1, 0, 0, 0, 0 ]
    initial_config = qnb.fermionic.tools.my_state( initial_config_occ_number, nbody_basis)

    # Then we build the excitation operator
    Excitation_op = (a_dagger_a[4,2] + a_dagger_a[5,3]) / np.sqrt(2)

    # We apply the excitation on the intial state and store it into a Psi WFT
    Psi = Excitation_op  @ initial_config

    # We visualize the final wavefunction
    qnb.fermionic.tools.visualize_wft(Psi,nbody_basis)
    print()


.. parsed-literal::


    	-----------
    	 Coeff.      N-body state
    	-------     -------------
    	-0.70711	|11011000⟩
    	+0.70711	|11100100⟩
