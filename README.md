<div align="center">
   
[<img src="logo.png" width="700">](https://quantnbody.readthedocs.io/en/latest/) 
   
</div>
    


[![DOI](https://joss.theoj.org/papers/10.21105/joss.04759/status.svg)](https://doi.org/10.21105/joss.04759) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## QuantNBody :  a python package for quantum chemistry/physics to manipulate many-body operators and wave functions.






<div align="center">
  
[<img src="button_documentation.svg" width="250"> ](https://quantnbody.readthedocs.io/en/latest/)  
                                                                                        
</div>


QuantNBody is a python package facilitating the implementation and manipulation of quantum many-body systems
composed of fermions or bosons.
It provides a quick and easy way to build many-body operators and wavefunctions and get access
(in a few python lines) to quantities/objects of interest for theoretical research and method developments. This tool can be also of a great help for pedagogical purpose and to help illustrate numerical methods for fermionic or bosonic systems. 

 

We provide below a non-exhaustive list of the various possibilities offered by the package:

- Visualizing the structure of any wavefunction in a given many-body basis (for fermionic and bosonic systems)
- Building 1-body, 2-body (...) reduced density matrices (for fermionic and bosonic systems)
- Building Spin operators $S^2$, $S_Z$, $S_+$  expressed in a many-body basis (for fermionic system)
- Building model Hamiltonians e.g. Bose-Hubbard, Fermi-Hubbard ( parameters given by the user )
- Building molecular *ab initio* Hamiltonians (needs psi4 to provide the electronic integrals)
- ...

To illustrate how to use this package, several example codes and tutorials have been implemented 
to help the new users (see the ''Tutorials'' folder).
Particularly, we show how to employ the tools already implemented to 
develop and implement famous many-body methods such as :
- FCI : Full Configuration Interaction (for bosonic and fermionic systems)
- CAS-CI : Complete Active Space CI  (for fermionic systems)
- SA-CASSCF : State-Averaged CAS Self-Consistent Field with orbital optimization (for fermionic systems)
- ...

 
--- 

 ## Installing the package (in development mode)
To install the latest version of QuantNBody in a quick and easy way:

```
git clone https://github.com/SYalouz/QuantNBody.git
cd QuantNBody
python -m pip install -e .
```
 
Note that you'll need to install the Psi4 package before installing QuantNBody. For this we redirect the user to the following link:
 
 - Psi4 installations : [Using conda](https://anaconda.org/psi4/psi4), see also the [following link](https://psicode.org/psi4manual/1.2.1/conda.html)

Once the package is fully installed, you can run some tests to check if everything was correctly done. For this, go the the [testing folder](https://github.com/SYalouz/QuantNBody/tree/main/testing) and run the following line in your terminal:
```
python TESTS.py
```


 ## Tutorials
 
Different examples and tutorials are furnished in the [Tutorials repository](https://github.com/SYalouz/QuantNBody/tree/main/Tutorials) under the form of Jupyter notebooks or python scripts.  


 ## How to contribute


We'd love to accept your contributions and patches to QuantNBody. There are a few small guidelines you need to follow.  

All submissions require review. We use GitHub pull requests for this purpose. Consult GitHub Help for more information on using pull requests. Furthermore, please make sure your new code comes with documentation.


 ## Support
 
If you are having issues, please let us know by posting the issue on our Github issue tracker.


## How to cite

When using QuantNBody for research projects, please cite [our reference paper](https://doi.org/10.21105/joss.04759)  :

Yalouz et al., (2022). QuantNBody: a Python package for quantum chemistry and physics to build and manipulate many-body operators and wave functions..     Journal of Open Source Software, 7(80), 4759, https://doi.org/10.21105/joss.04759 

We are happy to include future contributors as authors on later releases. 




  
