---
title: 'QuantNBody:  a Python package for quantum chemistry and physics to build and manipulate many-body operators and wave functions.'
tags:
  - Python
  - quantum physics and chemistry
  - quantum many-body systems
  - exact diagonalization
authors:
  - name: Saad Yalouz^[yalouzsaad@gmail.com]
    orcid: 0000-0002-8818-3379
    affiliation: 1  # (Multiple affiliations must be quoted)
  - name: Martin Rafael Gullin # note this makes a footnote saying 'Co-first author'
    affiliation: 1
  - name: Sajanthan Sekaran
    affiliation: 1
affiliations:
 - name: Laboratoire de Chimie Quantique, Institut de Chimie, CNRS/Université de Strasbourg, 4 rue Blaise Pascal, 67000 Strasbourg, France
   index: 1
date: 11 March 2022
bibliography: paper.bib
 
---

# Summary

The 'QuantNBody' package is a Python toolkit for quantum chemists/physicists interested in methods development to study quantum many-body problems ranging from electronic structure to condensed matter theory. It provides a quick and easy way to build matrix representations of bosonic and fermionic quantum many-body operators (e.g., Hamiltonians, spin or excitation operators) and get access to quantities/objects of interest (e.g., energies, reduced density matrices, many-body wave functions). The code includes various native functions and it is flexible enough to help users in building their own numerical tools to implement new methods. 

# Statement of need

The manipulation of many-body operators in the language of second quantization is a crucial step for accessing the properties of model or *ab initio* systems in quantum chemistry and physics.  From a numerical point of view, this requires developing codes that can build matrix representations of quantum operators in a given quantum many-body basis (e.g., Hamiltonians, spin or excitation operators).  In a great majority of cases, this aspect is kept as a ''blackbox'' in codes to spare the users from cumbersome numerical parts and to facilitate the use of already implemented methods. Nevertheless, this type of implementation can appear as a real obstacle for researchers in need of reliable numerical tools to quickly develop and test new methodologies based on second quantization algebra. 

The 'QuantNBody' code was designed to answer this problem: it is a theoretician-friendly package which provides an efficient and simple-to-use numerical toolkit to help create and manipulate operators and wavefunctions related to quantum many-body systems (with either fermionic or bosonic particles). It allows straightforward implementations of second quantization into numerical objects (i.e., matrices) in order to facilitate the conversion of analytical developments into compact and easy to write numerical Python codes. 

The user-friendly philosophy of QuantNBody regarding second quantization encoding can be compared (to some extent) to the FermiLib module originally developped for OpenFermion [@mcclean2020openfermion] (a quantum computing oriented package). In practice, the advantage of QuantNBody lies in its particle number conserving encoding  which differs from what is done in OpenFermion (where the full Fockspace is encoded). As a result, QuantNBody allows the realization of larger scale calculations (up to 8 electrons in 8 spatial molecular orbitals in a few seconds). This new toolkit opens new perspectives for  many-body simulations in Python and then completes the panel of already existing theoretician-friendly packages dedicated to many-body systems such as Quimb [@gray2018quimb], Quspin [@weinberg2017quspin;@weinberg2019quspin], QuTIP [@johansson2012qutip;@johansson2013qutip], Yao [@luo2020yao] and OpenFermion [@mcclean2020openfermion].  


# Framework of the package

The QuantNbody package employs the SciPy library [@2020SciPy-NMeth] for the creation of sparse matrices representation of many-body operators. The Numba package [@lam2015numba] is also used to accelerate calculation when possible. The framework of the package lies in two fundamental ingredients. The first one is the creation of a reference many-body vector basis (based on a total number of quantum particles and modes/orbitals to fill) in which second quantization operators can be represented. The second ingredient consists in creating a general tool that can help build any particle-number conserving many-body operator which is: the single-body hopping operators $a^\dagger_p a_q$. Once these two ingredients have been created, the user can employ various pre-built functions in order to (i) construct different types of many-body operators (e.g., Hamiltonians, spin and excitation operators), (ii) manipulate/visualize quantum many-body states. All the native functions have been thought to facilitate calculations for new users and young students. Beyond this, the QuantNBody package has been also designed to provide flexibility to experimented users to develop their own tools to implement/test their own methods.

# A quick illustration

As an illustration of the various types of system one can treat with the QuantNBody package, we describe here a few native many-body Hamiltonians which have already been implemented for fermionic and bosonic problems.

## Fermionic systems
Different fermionic Hamiltonians can be straightforwardly built with the QuantNBody package. For example, a pre-built function of the package allows a quick implementation of the quantum chemistry *ab initio* molecular electronic structure Hamiltonian
\begin{equation} \label{eq:mol}
\hat{H}= \sum_{p,q} h_{pq} \sum_\sigma^{\uparrow,\downarrow} a^\dagger_{p,\sigma} a_{q,\sigma} 
+ \frac{1}{2} \sum_{p,q,r,s}  g_{pqrs} \sum_{\sigma,\tau}^{\uparrow,\downarrow} a^\dagger_{p,\sigma} a^\dagger_{r,\tau} a_{s,\tau} a_{q,\sigma}  ,
\end{equation}
where $a^\dagger_{i,\sigma}$ ($a_{i,\sigma}$) are the fermionic creation (annihilation) operators in the orbital $i$ with spin $\sigma$. In a similar way,  another native function has been also created to numerically build the Fermi-Hubbard Hamiltonian from condensed matter theory
\begin{equation} \label{eq:FH}
\hat{H}_{FH} = -t  \sum_{i,j} \sum_\sigma^{\uparrow,\downarrow} a^\dagger_{i,\sigma} a_{j,\sigma} 
+ U \sum_{i}  a^\dagger_{i,\uparrow}a_{i,\uparrow} a^\dagger_{i,\downarrow} a_{i,\downarrow},
\end{equation}
In practice, native functions are already implemented in the QuantNBody package to build both types of fermionic Hamiltonians. This implementation is based on the use of the already built single-body hopping operators mentioned earlier to generate the one- and two-body fermionic excitation operators (present in Eq. \ref{eq:FH} and Eq. \ref{eq:mol}). Nevertheless, the one-/two-body integrals (i.e., $h_{pq}$, $g_{pqrs}$ and $t$ and $U$) have to be defined by the user.
They can be set as pure parameters or obtained from external chemistry Python packages like PySCF [@sun2020recent] or Psi4 [@parrish2017psi4].
As an illustration, we show in Fig. 1 results one can produce with the package for both fermionic Hamiltonians. We focus here on the calculation of the ground state (noted $| \Psi_0\rangle$) in a H$_2$ molecule and a Fermi-Hubbard dimer, and evaluate several associated properties (energy and many-body basis decomposition).
 
![H$_2$ molecule and Fermi-Hubbard dimer. **Left column:** ground state energy and decomposition in the many-body basis for the dissociation of the H$_2$ molecule in a minimal basis (STO-3G) using integrals from Psi4 [@parrish2017psi4]. **Right column:** similar properties for the Fermi-Hubbard dimer as a function of $U/t$ (2 electrons on 2 sites and $t = 1$). ](figure_fermion.png)

## Bosonic systems
 
Bosonic Hamiltonians can also be numerically built with the QuantNBody package. For example, a pre-built function allows a quick implementation of the Bose-Hubbard Hamiltonian
\begin{equation} 
\hat{H}_{BH} = -t  \sum_{i,j}   a^\dagger_{i} a_{j} 
+ U \sum_{i}  a^\dagger_{i} a_{i}  ( a^\dagger_{i} a_{i}  -1 ) ,
\end{equation}
where $a^\dagger_{i}$ ($a_{i}$) are now the bosonic creation (annihilation) operators of the local site $i$.
Here again, the native function implementing the Bose-Hubbard Hamiltonian in the QuantNBody package manages on its own the building of all the one- and two-body bosonic operators via the already built single-particle hopping operators. The one-/two-body integrals (i.e., $t$ and $U$) have to be defined by the user. As an illustration, we present in Fig. 2 the ground state properties one can obtain using the native function that implements the Bose-Hubbard Hamiltonian (two bosons on two sites). 

 
![Bose-Hubbard dimer with two bosons. **Left column:** illustration of the system. **Right column:** ground state energy and its decomposition in the many-body basis for the Bose-Hubbard dimer as a function of $U/t$ (with $t = 1$).](figure_boson.png)
 
# Related projects

The QuantNBody package is being currently used in the "Laboratoire de Chimie Quantique de Strasbourg" in several projects dedicated to strongly correlated systems. These projects including the study of spin properties in metal-ligand molecular systems [@roseiro2022excited], the development of embedding methods for large fermionic systems [@yalouz2022quantum] and the development of orbital optimisation techniques for fermionic excited states  [@yalouz2022orthogonally] to cite but a few. As future developments, we plan to extend the capacities of the package to the treatment of hybrid systems mixing both fermions and bosons degrees of freedom (e.g., for polaritonic chemistry or exciton-phonon systems).

# Acknowledgements

Saad Yalouz acknowledges support from the Interdisciplinary Thematic Institute ITI-CSC
via the IdEx Unistra (ANR-10-IDEX-0002) within the program Investissement d’Avenir.

# References

