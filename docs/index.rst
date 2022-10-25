QuantNBody
===========

QuantNBody : QuantNBody : a python package for quantum chemistry/physics to manipulate many-body operators and wave functions.


QuantNBody is a python package facilitating the implementation and manipulation of quantum many-body systems
composed of fermions or bosons. It provides a quick and easy way to build many-body operators and wavefunctions and get access
(in a few python lines) to quantities/objects of interest for theoretical research and method developements. This tool can be also of a great help for pedagogical purpose and to help illustrating numerical methods for fermionic or bosonic systems

::
  
  python SlowQuant.py MOLECULE SETTINGS
  
As a ready to run example:

::
  
  python SlowQuant.py H2O.csv settingExample.csv

QuantNBody have the following requirements:

- Python 3.5 or above
- numpy 1.13.1 
- scipy 0.19.1  
- numba 0.34.0
- cython 0.25.2
- gcc 5.4.0

.. toctree::
   :maxdepth: 2
   :caption: How to use
   
   install.rst
   keywords.rst
   Examples.rst
   issues.rst
   illustrativecalc.rst

.. toctree::
   :maxdepth: 2
   :caption: Working equations and functions 
   
   General.rst
   MolecularIntegral.rst
   IntegralTrans.rst
   HFMethods.rst
   DIIS.rst
   MPn.rst
   Properties.rst
   GeoOpt.rst
   CI.rst
   CC.rst
   BOMD.rst
   
