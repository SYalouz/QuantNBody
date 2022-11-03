.. QuantNBody documentation master file, created by
   sphinx-quickstart on Sat Oct 29 14:31:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QuantNBody
==========

**QuantNBody** is a python package facilitating the implementation and manipulation of quantum many-body systems
composed of fermions or bosons.
It provides a quick and easy way to build many-body operators and wavefunctions and get access
(in a few python lines) to quantities and objects of interest for theoretical research and method developements.
This tool can be also of a great help for pedagogical purpose and to help illustrating numerical methods for fermionic or bosonic systems.
We provide below a non-exhaustive list of the various possibilites offered by the package:

*  Visualizing the structure of any wavefunction in a given many-body basis (for fermionic and bosonic systems)
*  Building 1-body, 2-body (...) reduced density matrices (for fermionic and bosonic systems)
*  Building Spin operators :math:`\hat{S}^2, \hat{S}_Z, \hat{S}_+`  expressed in a many-body basis (for fermionic system)
*  Building model Hamiltonians :math:`\hat{H}` (*e.g.* Bose-Hubbard, Fermi-Hubbard)
*  Building molecular *ab initio* Hamiltonians :math:`\hat{H}` (psi4 provides the electronic integrals)

To illustrate how to use this package, several example codes and tutorials have been implemented (see ''Tutorials and examples'').


.. toctree::
   :maxdepth: 4
   :caption: How to install

   install.rst

.. toctree::
   :maxdepth: 4
   :caption: Tutorials and examples

   remarks.rst
   Tuto_FIRST_STEP.rst
   Tuto_PLAYING_WITH_STATES.rst
   Tuto_SPIN_AND_AB_INITIO_HAMILTONIANS.rst
   Tuto_BOSE_HUBBARD.rst

.. toctree::
   :maxdepth: 4
   :caption: List of functions implemented

   remarks_structure.rst
   api.rst

.. toctree::
   :maxdepth: 4
   :caption: Data structures



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
