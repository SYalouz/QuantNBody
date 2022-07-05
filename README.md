
![](logo.png)

# QuantNBody : a python package for quantum chemistry/physics to manipulate many-body operators and wave functions.
 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6565736.svg)](https://doi.org/10.5281/zenodo.6565736)

QuantNBody is dedicated to the manipulation of quantum many-body systems composed either of electron or bosons.
It provides a quick and easy way to build many-body operators and wavefunctions and get access
(in a few python lines) to quantities/objects such as :

- The decomposition of any wavefunction in a given many-body basis
- 1-body, 2-body (...) reduced density matrices 
- Spin operators S^2, S_z (for fermionic system) expressed in a many-body basis
- Already made model Hamiltonians e.g. Bose-Hubbard, Fermi-Hubbar
- Already made molecular ab initio Hamiltonians (requires a connexion to Psi4 or PySCF)
- ...

 # Quick and easy installation (Developement mode)
To install the latest version of QuantNBody:

```
git clone https://github.com/SYalouz/QuantNBody.git

cd QuantNBody

python -m pip install -e .
```

Hamiltonians already implemented:

- Ab initio electronic structure Hamiltonian (needs psi4 or PySCf to provide the electronic integrals)
- Fermi-Hubbard molecules Hamiltonians ( parameters given by the user )
- Bose-Hubbard Hamiltonians ( parameters given by the user )

Methods one can implement from scratch with the tools from QuantNBody (see the Example folder):
- FCI : Full Configuration Interaction
- CAS-CI : Complete Active Space CI  
- SA-CASSCF : State-Averaged  CAS Self-Consistent Field (with orbital optimization)
- ...
