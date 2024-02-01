Hybrid Fermion-Boson Systems in QuantNBody
==========================================

Lucie Pepe - - Laboratoire de Chimie Quantique de Strasbourg, France -
January 2024

Hybrid quantum many-body systems are prevalent in nature and they
originate from the intricate interplay between electrons and bosons.
These systems manifest, for example, when a molecular systems (or
materials) interact with an external environment, incorporating photons
and/or phonons. This is the case for exemple in polaritonic quantum
chemistry (i.e., quantum electrodynamics), where the electronic
structure of a molecule interacts with the photonic field of a cavity.
Similarly, in condensed matter physics, the electronic degrees of
freedom (or excitonic ones) experience perturbation owing to the
presence of a vibrational phononic environment. From a theoretical point
of view, the total number of electrons :math:`N_{elec}` in these systems
is conserved as constant, but this is not the case for the bosonic
number :math:`N_{bos}`.



The QuantNBody package provides all the necessary tools to simulate such
hybrid quantum many-body systems. In the subsequent sections, we will
provide a comprehensive, step-by-step breakdown of its functionality.
Before starting, let us import the package ;-)

.. code:: ipython3

    # Import the quantnbody package
    import quantnbody as qnb
    import numpy as np

Step 1: Building a hybrid many-body basis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hybrid quantum many-body systems can formally be described by a hybrid
Hilbert space :math:`\mathcal{H}_{hyb}` composed of a set of
electron-boson configurations
:math:`\lbrace | \Phi_{hyb}^\kappa\rangle \rbrace_{\kappa=1}^{ \dim(\mathcal{H}_{hyb})}`
expressed as:

.. math::  | \Phi_{hyb}^\kappa \rangle = | \Phi_{bos} \rangle \otimes | \Phi_{elec} \rangle. 

Here, the bosonic configuration :math:`| \Psi_{bos} \rangle` describes
how :math:`N_{bos}` bosons occupy :math:`N_{mode}` modes, while the
electronic configuration :math:`| \Psi_{elec} \rangle` is a Slater
determinant describing how :math:`N_{elec}` electrons occupy a
collection of :math:`2 N_{MO}` spin-orbitals (i.e. :math:`N_{MO}`
spatial orbitals).

The total number of accessible fermion-boson configurations
:math:`\lbrace | \Phi_{hyb}^\kappa\rangle \rbrace_{\kappa=1}^{ \dim(\mathcal{H}_{hyb})}`
is given by the dimension of the hybrid Hilbert space:

.. math:: \dim({\mathcal{H}_{hyb}}) =  \dim(\mathcal{H}_{bos}) \times \dim(\mathcal{H}_{elec}), 

with

.. math:: \dim({\mathcal{H}_{bos}}) = \sum_{N_{bos}=0}^{N_{bos}^{MAX}}\binom{N_{bos} + N_{mode} - 1 }{N_{mode}} \quad\quad \text{  AND} \quad\quad \dim({\mathcal{H}_{elec}}) = \binom{2N_{MO}}{N_{elec}} . 

- :math:`\dim({\mathcal{H}_{bos}})` describes the dimension of the
bosonic Fock-space. It includes all the possible distributions of
:math:`N_{bos}` bosons in :math:`N_{modes}` modes (with $N_b = 0
:raw-latex:`\rightarrow `N\_{b}^{max} $). Note again that this space
doesn’t preserve the total number of bosons! -
:math:`\dim({\mathcal{H}_{elec}})` describes the electronic space
counting all the possibilites to distribute :math:`N_{elec}` in
:math:`N_{MO}` spatial orbitals.

**How to create such a many-body basis with QuantNBody ?**

The QuantNBody package builds a numerical representation of such a
hybrid many-body basis as a list of states describing the repartition of
:math:`N_{elec}` electrons in :math:`2N_{MO}` spin-orbitals, combined
with the repartition of :math:`N_{b}` bosons in :math:`N_{modes}` modes.
These states are numerically referenced by a list of kappa indices such
that :

.. math::


   \Big\lbrace |\kappa \rangle \Big\rbrace_{\textstyle \kappa=1}^{\textstyle \dim_{\mathcal{H}_{bos/elec}}}

**A little example with :math:`N_{MO}=N_{elec}=2` and
:math:`N_{bos}=N_{modes}=2`: in this case, we should have 36 many-body
states**

.. code:: ipython3

    # ======================================
    # Define the fermionic system
    # ======================================
    N_elec = 2 # number of fermions 
    N_MO   = 2 # number of molecular orbitals 
    
    # ======================================
    # Define the bosonic system
    # ======================================
    N_b_max = 2 # maximal number of bosons in the whole system 
    N_mode  = 2 # number of bosonic modes 
    list_bosons = range(N_b_max+1) # list of all possible number of bosons that can be distributed in the bosonic modes  
    
    # ======================================
    # Build the hybrid many-body basis
    # ======================================
    nbody_basis = qnb.hybrid_fermionic_bosonic.tools.build_nbody_basis(N_mode, list_bosons, N_MO, N_elec) 
    
    # Print results
    print('Shape  of the hybrid kappa states')
    for s in range(len(nbody_basis)):
        print('| kappa={} >'.format(s), '=', nbody_basis[s])


.. parsed-literal::

    Shape  of the hybrid kappa states
    | kappa=0 > = [0 0 1 1 0 0]
    | kappa=1 > = [0 0 1 0 1 0]
    | kappa=2 > = [0 0 1 0 0 1]
    | kappa=3 > = [0 0 0 1 1 0]
    | kappa=4 > = [0 0 0 1 0 1]
    | kappa=5 > = [0 0 0 0 1 1]
    | kappa=6 > = [1 0 1 1 0 0]
    | kappa=7 > = [1 0 1 0 1 0]
    | kappa=8 > = [1 0 1 0 0 1]
    | kappa=9 > = [1 0 0 1 1 0]
    | kappa=10 > = [1 0 0 1 0 1]
    | kappa=11 > = [1 0 0 0 1 1]
    | kappa=12 > = [0 1 1 1 0 0]
    | kappa=13 > = [0 1 1 0 1 0]
    | kappa=14 > = [0 1 1 0 0 1]
    | kappa=15 > = [0 1 0 1 1 0]
    | kappa=16 > = [0 1 0 1 0 1]
    | kappa=17 > = [0 1 0 0 1 1]
    | kappa=18 > = [2 0 1 1 0 0]
    | kappa=19 > = [2 0 1 0 1 0]
    | kappa=20 > = [2 0 1 0 0 1]
    | kappa=21 > = [2 0 0 1 1 0]
    | kappa=22 > = [2 0 0 1 0 1]
    | kappa=23 > = [2 0 0 0 1 1]
    | kappa=24 > = [1 1 1 1 0 0]
    | kappa=25 > = [1 1 1 0 1 0]
    | kappa=26 > = [1 1 1 0 0 1]
    | kappa=27 > = [1 1 0 1 1 0]
    | kappa=28 > = [1 1 0 1 0 1]
    | kappa=29 > = [1 1 0 0 1 1]
    | kappa=30 > = [0 2 1 1 0 0]
    | kappa=31 > = [0 2 1 0 1 0]
    | kappa=32 > = [0 2 1 0 0 1]
    | kappa=33 > = [0 2 0 1 1 0]
    | kappa=34 > = [0 2 0 1 0 1]
    | kappa=35 > = [0 2 0 0 1 1]


**What is the meaning of these 36 states ?**

Here, each list of number string represents an hybrid many-body
occupation number state. As an example, let’s check the first state for
which we have :

.. math:: | \kappa  = 0\rangle = | \underbrace{0}_{\substack{\textstyle{ 1st }\\ \textstyle{ mode}}}, \; \; \;\underbrace{0}_{\substack{\textstyle{ 2nd}\\ \textstyle{ mode}}},\;\underbrace{   \overbrace{1}^{ \textstyle  {\alpha}}, \; \; \;\overbrace{1}^{ \textstyle  {\beta}},}_{\textstyle 1st \ MO}\; \; \underbrace{\overbrace{0}^{ \textstyle  {\alpha}}, \; \; \; \overbrace{0}^{ \textstyle  {\beta}}}_{\textstyle 2nd \ MO} \rangle

Here we choose to structure the occupation numbers as follows:

-  Bosonic modes are expressed at the beginning of the list of numbers.
   Each value refers to the number of bosons in the associated bosonic
   mode.
-  For the following fermionic part, each couple of terms refer to **a
   same spatial orbital**, with an alternation of **:math:`\alpha`-**
   and **:math:`\beta`-spinorbitals**.

Considering the 36 states contained in the list, we see that the first
set of 6 lines contains all the possible fermionic configurations, for a
vacuum bosonic configuration. Then, the next following states describe
all possible fermionic configurations, for another bosonic configuration
where we consider 1 boson in the two modes, and so on… until all the
possible repartitions of the $N\_{bos} = 0
:raw-latex:`\rightarrow `N\_{bos}^{max} $ bosons in the $ N\_{mode}$
modes have been scanned.

Step 2: About building operators in the hybrid many-body basis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In QuantNbody, to treat hybrid systems two types of many-body operators
are considered as elementary bricks:

**Fermionic hopping operators :math:`\hat{a}^\dagger \hat{a}`:** indeed,
every operator conserving the total number of fermionic particles can be
decomposed as a serie of electronic hopping operators
:math:`\hat{a}^\dagger \hat{a} \otimes \mathbb{1}_{bos}` which only act
on the fermionic part leave unchanged the bosonic part of the
hybrid-states.

**Bosonic creation/anihilation :math:`\hat{b}^\dagger/\hat{b}`:**
indeed, every operator not conserving the total number of bosonic
particles may be expressed as a series of anihilation (or creation)
bosonic operators $ :raw-latex:`\hat{b}`
:raw-latex:`\otimes `:raw-latex:`\mathbb{1}`\_{elec}$ that only act on
the bosonic part of the states and leave the fermionic part unchanged.

The QuantNbody package provides a matrix representation of these two
central operators in the numerical hybrid-many body basis. If we
generically call the latter operators by :math:`\hat{O}`, this means in
practice that we create a matrix representation such that

.. math::

    \hat{O} = \sum_{\kappa, \kappa' 
    =1}^{\dim(\mathcal{H}_{hyb})}  \langle \kappa' | \hat{O} | \kappa  \rangle  \; | \kappa'    \rangle\langle \kappa |  

Step 3: Build the fermionic :math:`\hat{a}^\dagger \hat{a}` operator in the hybrid basis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the hybrid part of the QuanNBody package, the fermionic operators
:math:`\hat{a}^\dagger_{p,\sigma} \hat{a}_{q,\tau}` can be built with a
simple command line:

.. code:: ipython3

    a_dagger_a = qnb.hybrid_fermionic_bosonic.tools.build_fermion_operator_a_dagger_a(nbody_basis, N_mode)

In practice, the way “a_dagger_a” will store each operator
:math:`\hat{a}^\dagger_{p,\sigma} \hat{a}_{q,\tau}` follows the way we
order the spin-orbitals in our many-body states. As an illustrative
example, taking the following elements will return the associated
many-body operators:

.. raw:: html

   <center>

a_dagger_a[0,0] $
:raw-latex:`\longrightarrow `:raw-latex:`\hat{a}`^:raw-latex:`\dagger`\ *{0,:raw-latex:`\alpha`}
:raw-latex:`\hat{a}`*\ {0,:raw-latex:`\alpha`}$

.. raw:: html

   </center>

.. raw:: html

   <center>

a_dagger_a[1,0] $
:raw-latex:`\longrightarrow `:raw-latex:`\hat{a}`^:raw-latex:`\dagger`\ *{0,:raw-latex:`\beta`}
:raw-latex:`\hat{a}`*\ {0,:raw-latex:`\alpha`}$

.. raw:: html

   </center>

.. raw:: html

   <center>

a_dagger_a[10,1] $
:raw-latex:`\longrightarrow `:raw-latex:`\hat{a}`^:raw-latex:`\dagger`\ *{5,:raw-latex:`\alpha`}
:raw-latex:`\hat{a}`*\ {0,:raw-latex:`\beta`}$

.. raw:: html

   </center>

**Example of the matrix shape of the hopping operator
:math:`\hat{a}^\dagger_{0,\alpha} \hat{a}_{1,\alpha}`:**

If we look at the element a_dagger_a[0,2], we get access to a sparse
matrix representation of the fermionic operator
:math:`\hat{a}^\dagger_{0,\alpha} \hat{a}_{1,\alpha}` in the hybrid
many-body basis which encodes the promotion of 1 electron from the the
2nd spin orbital (second MO, spin up) to the oth spin orbital (first MO,
spin up) of the fermionic sub-system:

.. code:: ipython3

    print(a_dagger_a[0,2])


.. parsed-literal::

      (0, 3)	-1.0
      (2, 5)	1.0
      (6, 9)	-1.0
      (8, 11)	1.0
      (12, 15)	-1.0
      (14, 17)	1.0
      (18, 21)	-1.0
      (20, 23)	1.0
      (24, 27)	-1.0
      (26, 29)	1.0
      (30, 33)	-1.0
      (32, 35)	1.0


We observe here that the action of this operator is only possible
between specific configurations. As an exemple, let us consider the
first line that shows a connexion between the $ :raw-latex:`\kappa  `$
states $|0 :raw-latex:`\rangle  `:raw-latex:`\leftrightarrow `\| 3
:raw-latex:`\rangle `$. These two states are actually given by \|
kappa=0 > = [0 0 1 1 0 0] and \| kappa=3 > = [0 0 0 1 1 0]. Here, we
clearly see that the action of the operator is well encoded:

-  The electron hops between the 0th and the 2nd spin-orbitals.
-  There is no change in the occupation number of the bosonic modes
   between these two states.

Step 3: Build the bosonic :math:`\hat{b}` and :math:`\hat{b}^\dagger` operators in the hybrid basis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the hybrid part of the QuanNBody package, the bosonic anihilation
operators :math:`\hat{b}_p` can be built with a simple command line:

.. code:: ipython3

    # We compute here the b operator
    b = qnb.hybrid_fermionic_bosonic.tools.build_boson_anihilation_operator_b(nbody_basis,N_mode)

In practice, the way “b” will store each operator :math:`\hat{b}_{p}`
follows the way we order the modes in our many-body states. As an
illustrative example, taking the following elements will return the
associated many-body operators:

.. raw:: html

   <center>

b[p] $ :raw-latex:`\longrightarrow `:raw-latex:`\hat{b}`\_{p} $

.. raw:: html

   </center>

Note that we can easily build the associated creation operator by taking
the tranposed version of each element such that

.. raw:: html

   <center>

b[p].T $
:raw-latex:`\longrightarrow `:raw-latex:`\hat{b}`\_{p}^:raw-latex:`\dagger  `$

.. raw:: html

   </center>

**Example of a bosonic anihilation operator :math:`\hat{b}_0`:**

If we look at the element b[0], we get access to a sparse matrix
representation of the bosonic anihilation operator :math:`\hat{b}_0` in
the hybrid many-body basis which encodes the desctruction of 1 boson in
the 0th mode:

.. code:: ipython3

    print(b[0])


.. parsed-literal::

      (0, 6)	1.0
      (1, 7)	1.0
      (2, 8)	1.0
      (3, 9)	1.0
      (4, 10)	1.0
      (5, 11)	1.0
      (6, 18)	1.4142135623730951
      (7, 19)	1.4142135623730951
      (8, 20)	1.4142135623730951
      (9, 21)	1.4142135623730951
      (10, 22)	1.4142135623730951
      (11, 23)	1.4142135623730951
      (12, 24)	1.0
      (13, 25)	1.0
      (14, 26)	1.0
      (15, 27)	1.0
      (16, 28)	1.0
      (17, 29)	1.0


We observe here that the action of this operator is only possible
between specific configurations. As an exemple, let us consider the
first element that shows a connexion between the $
:raw-latex:`\kappa  `$ states $|0
:raw-latex:`\rangle  `:raw-latex:`\leftrightarrow `\| 6
:raw-latex:`\rangle `$. These two states are actually given by \|
kappa=0 > = [0 0 1 1 0 0] and \| kappa=6 > = [1 0 1 1 0 0]. Here, we
clearly see that the action of the operator is well encoded:

-  The two states are related by the creation/anhihilation of one boson
   in the 0th mode.
-  There is no change in the fermionic occupation numbers of the
   spin-orbitals between the two states.

**Last exemple with a counting :math:`\hat{b}_1^\dagger \hat{b}_1`
operator:**

Once all the :math:`\hat{b}_p` are built, one can use these operators as
building blocks for a wide possibilty of operators such as the
:math:`\hat{n}_p = \hat{b}^\dagger_p \hat{b}_p` counting one. As an
exemple, let’s count the number of bosons in the second mode of the
following state we want to target

.. math::    | \Phi_{bos} \rangle \otimes | \Phi_{elec} \rangle = |02\rangle \otimes |1100 \rangle

QuantNBody provides a way to build our own state from a given occupation
number list as follows

.. code:: ipython3

    # 1) Define the occupation number list of bosonic modes and fermionic spin-orbitals
    LIST_OCC_NUMB = [0,2,1,1,0,0]
    
    # 2) Obtain the qnb traduction in the hybrid many-body basis   
    my_many_body_state =  qnb.hybrid_fermionic_bosonic.tools.my_state(LIST_OCC_NUMB, nbody_basis)
    
    # 2) Visualize the associated wavefunction 
    print( 'Initial state :')
    qnb.hybrid_fermionic_bosonic.tools.visualize_wft(my_many_body_state,
                                                     nbody_basis, 
                                                     N_mode )
    print()


.. parsed-literal::

    Initial state :
    
    	-----------
    	 Coeff.     N-body state and index 
    	-------     ----------------------
    	+1.00000   |02⟩ ⊗ |1100⟩    #30 
    
    


Let us now count the number of bosons in this state as follows:

.. code:: ipython3

    n_1 = b[1].T@b[1]
    print("Total number of boson in the targeted state\n", my_many_body_state.T @ n_1 @ my_many_body_state )


.. parsed-literal::

    Total number of boson in the targeted state
     2.0000000000000004

