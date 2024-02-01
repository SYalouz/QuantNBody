A few remarks
=============


In the following **QuantNBody tutorials**, we will detail how the package can be used in practice to manipulate and create many-body operators and wavefunctions for fermionic and bosonic systems.
We encourage the new users to follow the tutorials in the following order:

*Tuto 1:* first steps with the package
  This tutorial explains the basics of the QuantNBody package. Focusing on fermionic systems,
  we explain how the encoding of a many-body basis  is realized in practice in the code (spin orbitals occupied by electrons).
  We also detail how we encode the single-body hopping operators. The latter being a central tool for the creation of any particle number conserving operators later on.

*Tuto 2:* playing with many-body wavefunctions
  This second tutorial illustrates how to easily manipulate many-body states. Here we detail how to create our own state (step by step) with native functions from QuantNBody. We also show how to apply excitation on these states in order to modify them and also how to visualize the resulting decomposition in the many-body basis.

*Tuto 3:* electronic structure Hamiltonian and spin operators
  This third tutorial focuses on the construction of different spin operators (*e.g.* :math:`\hat{S}_2`) and *ab initio* electronic structure Hamiltonians. We show here how to easily build both types of operator. Furthermore, we demonstrate how to combine these operators together to find specific eigenstates with a desired spin symmetry and compare the results to Psi4.

*Tuto 4:* the Bose-Hubbard system
  For those interested in bosonic systems, we also describe here equivalent features/functions to build operators, see and manipulate wavefunctions.

*Tuto 5:* definition of hybrid fermions_bosons sytems
  This fifth tutorial is an explanation about the (new !) extension of the QuantNBody package to the computation of an hybrid (fermions interacting with bosons) many-body basis. We also show how to encode fermionic and bosonic operators respectively, in this new hybrid many-body basis, and how to use them to visualize polaritonic states.

*Tuto 6:* Holstein and polaritronic QED dynamics
  This last tutorial is an application of the Tuto 4, where the new fermionic and bosonic operators (expressed in an hybrid many-body basis) are used as building blocks to encode both Holstein and polaritonic-QED Hamiltonians. Some observables, such as the ime-dependentpopulations of fermionic orbitals and bosonic modes (in the Holstein example) or optical spectra (in the QED system), are also computed.

**Enjoy the tutorials !**  |:wink:| |:+1:|

.. note::

  In addition to what is presented here,  we refer the interested user to our Github page where we provide a series of `Jupyter-notebooks
  and Python scripts <https://github.com/SYalouz/QuantNBody/tree/main/Tutorials>`_ illustrating different types of many-body calculations.
  All these methods are implemented from scratch with the QuantNBody packages.
  The folders are named according to the type of system used as a study case:

  - `/Examples_BoseHubbard/: <https://github.com/SYalouz/QuantNBody/tree/main/Tutorials/Examples_BoseHubbard>`_ contains script for FCI calculations on small-sized Bose-Hubbard systems.
  - `/Examples_FermiHubbard/: <https://github.com/SYalouz/QuantNBody/tree/main/Tutorials/Examples_FermiHubbard>`_ contains scripts for CAS-CI and FCI calculations on small-sized Fermi-Hubbard systems.
  - `/Examples_chemistry/: <https://github.com/SYalouz/QuantNBody/tree/main/Tutorials/Examples_chemistry>`_ contains scripts for CAS-CI, SA-CASSCF and FCI calculation on small-sized molecular systems.
