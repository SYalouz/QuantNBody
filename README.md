
![](logo.png)

# QuantNBody : a python package for quantum chemistry/physics to manipulate many-body operators and wave functions.
 
Please, if you are using this code, cite the following : [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6565736.svg)](https://doi.org/10.5281/zenodo.6565736)

QuantNBody is dedicated to the manipulation of quantum N-body systems ranging from many-electron to many-bosons hamiltonians.
It provides a quick and easy way to build many-body operators and get access
(in a few python lines) to important quantities/objects such as :
 
- The decomposition of a wavefunction in a many-body basis
- The 1-body, 2-body (...) reduced density matrices (espressed in the Molecular orbital or spinorbital basis)
- The spin operators S^2, S_z (for fermionic system expressed in a many-body basis)
- ...
 
Hamiltonians already implemented:

- Ab initio electronic structure Hamiltonian (needs psi4 or PySCf to provide the electronic integrals)
- Fermi-Hubbard molecules Hamiltonians ( parameters given by the user )
- Bose-Hubbard Hamiltonians ( parameters given by the user )

Methods one can implement from scratch with the tools from QuantNBody (see the Example folder):
- FCI : Full Configuration Interaction
- CAS-CI : Complete Active Space CI  
- SA-CASSCF : State-Averaged  CAS Self-Consistent Field (with orbital optimization)
- ...
