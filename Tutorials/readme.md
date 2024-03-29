# Welcome to the tutorials !

We gather a series of tutorials/examples detailing how to use the QuantNBody package for different types of many-body systems. 

## Notebooks tutorials

In order to domesticate the package, we encourage the new users to read the jupyter notebooks tutorials in the following order:

- *Tuto_FIRST_STEP.ipynb:* 
This tutorial teaches the basics of the QuantNBody package. Focusing on fermionic systems, 
we explain how the encoding of a many-body basis  is realized in practice in the code (spin orbitals occupied by electrons).
We also detail how we encode the single-body hopping operators. The latter being a central tool for the creation of any particle number conserving operators later on. 

- *Tuto_PLAYING_WITH_STATES.ipynb:* 
This second tutorial illustrates how to easily manipulate many-body states. Here we detail how to create our own state step by step with native functions from QuantNBody. We also show how to apply excitation on these states to modify them and also how to visualize the resulting decomposition in the many-body basis.

- *Tuto_SPIN_AND_AB_INITIO_HAMILTONIANS.ipynb:* 
This third tutorial focuses on the construction of different spin operators (*e.g.* $\hat{S}_2$) and *ab initio* electronic structure Hamiltonians. We show here how to easily build both types of operator. Furthermore, we demonstrate how to combine these operators together to find specific eigenstates with a desired spin symmetry and compare the results to Psi4.

- *Tuto_BOSE_HUBBARD.ipynb:* 
For those interesting in bosonic systems, we also describe here equivalent features/functions to build operators, see and manipulate wavefunctions.


## Examples of scripts

We also provide a series of python scripts illustrating different types of many-body calculations. All these methods are implemented from scratch with the QuantNBody. The folders are named according to the type of system used as a study case:

- *Examples_BoseHubbard:* script for FCI calculations on small-sized Bose-Hubbard systems.
- *Examples_FermiHubbard:* scripts for CAS-CI and FCI calculations on small-sized Fermi-Hubbard systems.
- *Examples_chemistry:* scripts for CAS-CI, SA-CASSCF and FCI calculation on small-sized molecular systems.
