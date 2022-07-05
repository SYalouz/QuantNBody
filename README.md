<p align="center">
  <img src="logo.png" width="700">  
</p> 
   
 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6565736.svg)](https://doi.org/10.5281/zenodo.6565736)

## QuantNBody : a python package to build and manipulate quantum many-body systems.


QuantNBody is a python package facilitating the implementation and manipulation of quantum many-body systems
composed of electron or bosons.
It provides a quick and easy way to build many-body operators and wavefunctions and get access
(in a few python lines) to quantities/objects of interest for research and method developements. Our goal here is to focus on resarch and teaching but not on highly efficient code for large simulations. We provide below a non-exhaustive list of the various possibilites offered by the package:

- Visualizing the structure of any wavefunction in a given many-body basis
- Building 1-body, 2-body (...) reduced density matrices 
- Building Spin operators $S^2$, $S_z$, $S_+$ (for fermionic system) expressed in a many-body basis
- Building model Hamiltonians e.g. Bose-Hubbard, Fermi-Hubbar ( parameters given by the user )
- Building molecular *ab initio* Hamiltonians (needs psi4 to provide the electronic integrals)
- ...

To illustrate how to use this package, several example codes and tutorials have been implemented 
to help the new users (see the ''Examples and Tutorials'' folder).
Particularily, we show how to employ the tools already implemented to 
develop and implement famous many-body methods such as :
- FCI : Full Configuration Interaction (for bosonic and fermionic systems)
- CAS-CI : Complete Active Space CI  (for fermionic systems)
- SA-CASSCF : State-Averaged CAS Self-Consistent Field with orbital optimization (fermionic systems)
- ...

Nota Bene:  

--- 

 ## How to easily install the package (in developement mode)
To install the latest version of QuantNBody in a quick and easy way:

```
git clone https://github.com/SYalouz/QuantNBody.git
cd QuantNBody
python -m pip install -e .
```
 
 Note that you'll need to install the Psi4 package before installing QuantNBody. For this we redirect the user to the following link:
 
 Psi4 installations :
 - [Using conda](https://anaconda.org/psi4/psi4), see also the [following link](https://psicode.org/psi4manual/1.2.1/conda.html)

