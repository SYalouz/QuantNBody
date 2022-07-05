<p align="center">
  <img src="logo.png" width="400">  
</p> 
  
 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6565736.svg)](https://doi.org/10.5281/zenodo.6565736)

## QuantNBody : a python package to build and manipulate quantum many-body systems.


QuantNBody is a python package facilitating the implementation and the manipulation of quantum many-body systems
composed either of electron or bosons.
It provides a quick and easy way to build many-body operators and wavefunctions and get access
(in a few python lines) to quantities/objects of interest for research and method developements. We provide below a non-exhaustive list of the various possibilites offered by the package

- Visualizing he decomposition of any wavefunction in a given many-body basis
- Building 1-body, 2-body (...) reduced density matrices 
- Building Spin operators S^2, S_z (for fermionic system) expressed in a many-body basis
- Building model Hamiltonians e.g. Bose-Hubbard, Fermi-Hubbar ( parameters given by the user )
- Building molecular ab initio Hamiltonians (needs psi4 or PySCf to provide the electronic integrals)
- ...

For example of the use of this package several tutorials have been implemented to help the new users.
Particularily, we provide illustrative code showing how to use the tools implemented to build their personal 
many-body methods such as (see the Example folder):
- FCI : Full Configuration Interaction (for bosonic and fermionic systems)
- CAS-CI : Complete Active Space CI  (for fermionic systems)
- SA-CASSCF : State-Averaged CAS Self-Consistent Field with orbital optimization (fermionic systems)
- ...

--- 

 ## How to easily install (in developement mode)
To install the latest version of QuantNBody in a quick and easy way:

```
git clone https://github.com/SYalouz/QuantNBody.git
cd QuantNBody
python -m pip install -e .
```
 

