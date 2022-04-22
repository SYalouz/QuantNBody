
![](logo.png)

# QuantNBody : a python package for quantum chemistry/physics to manipulate many-body operators and wave functions.

The goal of the package is to provide a quick and easy way to build many-body operators and get access
(in a few python lines) to important quantities/objects such as :
 
- The shape of a wavefunction in a many-body basis (the so called CI-coefficients)
- The 1-body, 2-body (...) reduced density matrices (espressed in the Molecular orbital or spinorbital basis)
- The spin operators S^2, S_z (expressed in a many-body basis)
- ...
 
 
Hamiltonians already implemented:

- Ab initio electronic structure Hamiltonian (needs psi4 or PySCf to provide the electronic integrals)
- Fermi-Hubbard molecules Hamiltonians ( parameters given by the user )


Particular transformation already implemented:

- Regular Householder transformation
- Block-Householder transformation 
 
