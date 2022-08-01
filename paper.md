---
title: 'QuantNBody:  a python package for quantum chemistry and physics to build and manipulate many-body operators and wave functions.'
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
 - name: Laboratoire de Chimie Quantique, 4 rue Blaise Pascal, 67000 Strasbourg, France
   index: 1
date: 11 March 2022
bibliography: paper.bib
 
---

# Summary

'QuantNBody' is a Python package providing numerical tools for quantum chemists/physicists interested in the development of methodologies to study quantum many-body problems ranging from electronic structure to condensed matter theory. It provides a quick and easy way to build matrix representations of bosonic and fermionic quantum many-body operators (e.g. hamiltonians, spin or excitation operators) and get access to quantities/objects of interest (e.g. energies, reduced density matrices, many-body wave functions). The code includes various native functions and it is flexible enough to help users in building their own numerical tools facilitating the implementation of new methods. 

# Statement of need

The numerical encoding of quantum many-body problems in the language of second quantization is a crucial tool for accessing the properties of model or ab initio systems in quantum chemistry and physics.  In practice, it requires developping a code that can build sparse matrix representations of different quantum operators in a given quantum many-body basis (e.g. Hamiltonian of a reference system, spin or excitation operators).  Usually, this aspect is kept as a ''blackbox'' to spare the user from cumbersome numerical parts. In practice, this type of package implementation suits perfectly for the realization of applications.
However, it is problematic for researchers who need numerical tools to develop and test new methodologies and theories quickly. 

The 'QuantNBody' Python package was designed with this goal in mind: to help theoreticians who need an easy way to numerically create and manipulate objects related to quantum many-body systems (with either fermionic or bosonic particles). The framework of the package lies in two fundamental ingredients. The first one is the creation of a reference many-body vector basis (based on a total number of quantum particles and modes/orbitals to fill) in which every particle-number conserving second quantization operator can be represented. The second ingredient consists in creating a general tool that can help build any particle-number conserving many-body operator : the single-body hopping operators $a^\dagger_p a_q$.  Once these two ingredients have been created, the user can employ various pre-built functions in order to (i) construct different types of many-body operators (e.g. hamiltonians, spin and excitation operators), (ii) manipulate/visualize quantum many-body states. All the native functions have been thought to facilitate calculations on many-body systems for new users and young students. Beyond this, the QuantNBody package has been also designed to provide more flexibility and to help more experimented users and researchers in developping their own tools to implement/test their own methods.

# A quick illustration

As an illustration of the various types of systems one can treat with the QuantNBody package, we here describe a few native many-body Hamiltonians which have already been implemented as native functions for both fermionic and bosonic problems.

## Fermionic systems
Different fermionic Hamiltonians can be straightforwardly built with the QuantNBody package. For example, a native function of the package allows a quick implementation of the quantum chemistry ab initio molecular electronic structure Hamiltonian
\begin{equation} 
\hat{H}= \sum_{p,q} h_{pq} \sum_\sigma^{\uparrow,\downarrow} a^\dagger_{p,\sigma} a_{q,\sigma} 
+ \frac{1}{2} \sum_{p,q,r,s}  g_{pqrs} \sum_{\sigma,\tau}^{\uparrow,\downarrow} a^\dagger_{p,\sigma} a^\dagger_{r,\tau} a_{s,\tau} a_{q,\sigma}  
\end{equation}
where $a^\dagger_{i,\sigma}$ ($a_{i,\sigma}$) are the fermionic creation (annihilation) operators for orbital $i$ with spin $\sigma$. Similarily, one can also use a native function to numerically build the Fermi-hubbard Hamiltonian from condensed matter theory
\begin{equation} 
\hat{H}_{FH} = -t  \sum_{i,j} \sum_\sigma^{\uparrow,\downarrow} a^\dagger_{i,\sigma} a_{j,\sigma} 
+ U \sum_{i}  a^\dagger_{i,\uparrow}a_{i,\uparrow} a^\dagger_{i,\downarrow} a_{i,\downarrow},
\end{equation}
The QuantNBody package manages on its own the building of all the one- and two-body fermionic operators via the already built object $a^\dagger a$ mentioned earlier. The one-/two-body integrals (i.e.  $h_{pq}$, $g_{pqrs}$  and $t$ and $U$ ) however have to be defined by the user.
They can be set as pure parameters or obtained from external chemistry python packages like PySCF [@sun2020recent] or Psi4 [@turney2012psi4].
As an illustration, we show in Fig. 1 a few results one can produce with the package for both fermionic hamiltonians. We focus here on the groundstate properties noted $| \Psi_0 \rangle$ (i.e. energy and many-body basis decomposition).
 
![$H_2$ molecule and Fermi-Hubbard dimer. **Left column:** ground state energy and ground state decomposition in the many-body basis for the $H_2$ molecule dissociation in a minimal basis (STO-3G) using integrals from Psi4 [@turney2012psi4]. **Right column:** ground state energy and ground state decomposition in the many-body basis for the Fermi-Hubbard dimer as a function of $U/t$  (2 electrons on 2 sites). \label{fig:example}](figure_fermion.png)

## Bosonic systems
 
Bosonic Hamiltonians can also be numerically built with the QuantNBody package. A native function of the package allows a quick implementation of the Bose-Hubbard Hamiltonian which reads
\begin{equation} 
\hat{H}_{BH} = -t  \sum_{i,j}   a^\dagger_{i} a_{j} 
+ U \sum_{i}  a^\dagger_{i} a_{i}  ( a^\dagger_{i} a_{i}  -1 ) 
\end{equation}
where $a^\dagger_{i}$ ($a_{i}$) are now the bosonic creation (annihilation) operators associated to a mode $i$.
Here again, the QuantNBody package manages on its own the building of all the one- and two-body bosonic operators via the already built object $a^\dagger a$ (which respect now bosonic algebra). The one-/two-body integrals (i.e.  $t$ and $U$ )  have to be defined by the user. As an illustration, we show in Fig. 2 the ground state properties one can obtain using the native function that implements a Bose-Hubbard Hamiltonian (two bosons on two sites). 

  
![Bose-Hubbard dimer with two bosons. **Left column:** illustration of the system. **Right column:** ground state energy and ground state decomposition in the many-body basis for the Bose-Hubbard dimer as a function of $U/t$. \label{fig:example}](figure_boson.png)
 
# Related projects

The QuantNBody package is being used currently in several projects conducted in the "Laboratoire de Chimie Quantique de Strasbourg" dedicated to strongly correlated systems. These projects range from the study of spin properties in metal-ligand molecular systems, the development of embedding methods for large fermionic systems, to the developement of new variational computational methods for bosonic systems to cite but a few.


# Acknowledgements

Saad Yalouz acknowledges support from the Interdisciplinary Thematic Institute ITI-CSC
via the IdEx Unistra (ANR-10-IDEX-0002) within the program Investissement d’Avenir.

# References

