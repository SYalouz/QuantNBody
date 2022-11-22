
.. figure:: logo_shadow.png
    :width: 800

What is QuantNBody ?
=====================

The **QuantNBody** package is a Python toolkit for quantum chemists/physicists interested in methods development to study
quantum many-body problems ranging from electronic structure to condensed matter theory. It provides a quick and easy way
to build matrix representations of bosonic and fermionic quantum many-body operators (*e.g.* Hamiltonians, spin or excitation
operators) and get access to quantities/objects of interest (*e.g.* energies, reduced density matrices, many-body wavefunctions).
The code includes various native functions and it is flexible enough to help users in building their own numerical tools to implement new methods.

.. note::

                        **QuantNBody: a package for whom ?** |:face_with_monocle:|

  .. figure:: research.png
     :scale: 35 %
     :alt: alternate text
     :align: left

  **A research-friendly package:** with QuantNBody, we want to facilitate the implementation of quantum many-body systems
  composed of fermions or bosons for theoretical research and method developments. It provides a quick and easy way to build many-body operators and wavefunctions and get access
  to quantities and objects of interest (in a few python lines).

  .. figure:: teaching.png
     :scale: 35 %
     :alt: alternate text
     :align: left

  **A teaching-friendly package:** with QuantNBody, we want to provide a tool to help teach theoretical concepts in chemistry and physics associated with quantum many-body systems.
  Indeed, we believe that this package can be useful for pedagogical purposes and to help illustrate numerical methods for fermionic or bosonic systems.

Framework of the package
===========================

The framework of the **QuantNbody** package lies in two fundamental ingredients.
The first one is the creation of a **reference many-body vector basis** (based on a total number of quantum particles and modes/orbitals to fill)
in which second quantization operators can be represented. The second ingredient consists in creating a general tool that can help build any
particle-number conserving many-body operator which is: **the single-body hopping operators** :math:`a^\dagger_p a_q`.  Once these two ingredients have been created,
the user can employ various pre-built functions in order to

  #. Construct different types of **many-body operators**.

  #. Manipulate/visualize **quantum many-body states**.

Beyond this, the QuantNBody package has also been designed to provide flexibility to experimented users to develop their own tools to implement/test their own methods.


Brief illustration of what is possible
=======================================

We provide below a non-exhaustive list of the various possibilities offered by the package:

  *  Visualizing the structure of any wavefunction in a given many-body basis (for fermionic and bosonic systems)
  *  Building 1-body, 2-body (...) reduced density matrices (for fermionic and bosonic systems)
  *  Building Spin operators :math:`\hat{S}^2, \hat{S}_Z, \hat{S}_+`  expressed in a many-body basis (for fermionic system)
  *  Building model Hamiltonians :math:`\hat{H}` (*e.g.* Bose-Hubbard, Fermi-Hubbard)
  *  Building molecular *ab initio* Hamiltonians :math:`\hat{H}` (Psi4 provides the electronic integrals)


Below we show some results one can produce with QuantNBody for the hydrogen molecule, a Fermi-Hubbard dimer and a Bose-Hubbard dimer.
To generate these data, the package was used to code (from scratch) each many-body Hamiltonian and to analyze their associated groundstate :math:`|\Psi_0\rangle`.

.. figure:: figure_fermion.png
    :width: 800

    **Left column:** Ground state energy and many-body decomposition of the hydrogen molecule groundstate in a minimal basis (STO-3G).
    **Right column:** similar properties for the Fermi-Hubbard dimer as a function of the on-site repulsion U (2 electrons on 2 sites).

.. figure:: figure_boson.png
    :width: 800

    **Left column:** Bose-Hubbard dimer with two bosons.  **Right column:** ground state energy and many-body decomposition
    for the Bose-Hubbard dimer as a function of the on-site repulsion U.

.. note ::

  To illustrate how to use the package, several tutorials are shown in the section ''Tutorials and examples''.


  How to contribute / Support
  ______________________________

  We'd love to accept your contributions and patches to QuantNBody. There are a few small guidelines you need to follow.

  All submissions require review. We use GitHub pull requests for this purpose. Consult GitHub Help for more information on using pull requests. Furthermore, please make sure your new code comes with documentation.

  If you are having issues, please let us know by posting the issue on our Github issue tracker.
