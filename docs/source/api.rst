Fermionic systems
=================

.. currentmodule:: quantnbody.fermionic.tools

In this subsection, we list a series of functions useful for fermionic systems.


_______________



Basic functions for the creation of many-fermion systems
----------------------------------------------------------------


.. autofunction:: build_nbody_basis


.. autofunction:: build_operator_a_dagger_a


_______________

_______________


Many-body Hamiltonians and excitations operators
--------------------------------------------------------
.. autofunction:: build_hamiltonian_quantum_chemistry


.. autofunction:: build_hamiltonian_fermi_hubbard


.. autofunction:: build_E_and_e_operators


_______________

_______________

Spin operators
---------------

.. autofunction:: build_s2_sz_splus_operator


.. autofunction:: build_s2_local


.. autofunction:: build_sAsB_coupling


.. autofunction:: build_spin_subspaces


_______________

_______________




Creating/manipulating/visualizing many-body wavefunctions
------------------------------------------------------------------------

.. autofunction:: my_state


.. autofunction:: visualize_wft


.. autofunction:: build_projector_active_space


.. autofunction:: weight_det


.. autofunction:: scalar_product_different_MO_basis


.. autofunction:: transform_psi_MO_basis1_in_MO_basis2


.. autofunction:: scalar_product_different_MO_basis_with_frozen_orbitals




_______________

_______________



Reduced density matrices
--------------------------------

.. autofunction:: build_1rdm_alpha


.. autofunction:: build_1rdm_beta


.. autofunction:: build_1rdm_spin_free


.. autofunction:: build_2rdm_fh_on_site_repulsion


.. autofunction:: build_2rdm_fh_dipolar_interactions


.. autofunction:: build_2rdm_spin_free


.. autofunction:: build_1rdm_and_2rdm_spin_free


.. autofunction:: build_hybrid_1rdm_alpha_beta


.. autofunction:: build_transition_1rdm_alpha


.. autofunction:: build_transition_1rdm_beta


.. autofunction:: build_transition_1rdm_spin_free


.. autofunction:: build_transition_2rdm_spin_free


.. autofunction:: build_full_mo_1rdm_and_2rdm_for_AS




_______________

_______________



Functions to manipulate fermionic integrals
------------------------------------------------

.. autofunction:: transform_1_2_body_tensors_in_new_basis


.. autofunction:: fh_get_active_space_integrals


.. autofunction:: fh_get_active_space_integrals_with_V


.. autofunction:: qc_get_active_space_integrals


_______________

_______________



Quantum embedding transformations (Householder)
--------------------------------------------------------

.. autofunction:: householder_transformation


.. autofunction:: block_householder_transformation



_______________

_______________



Psi4 calculation helper
------------------------

.. autofunction:: get_info_from_psi4


.. autofunction:: generate_h_chain_geometry


.. autofunction:: generate_h_ring_geometry


.. autofunction:: generate_h4_geometry

_______________

_______________


Orbital optimization
---------------------

.. note::
  The following functions have been developed for a specific application to the *ab initio* electronic structure problem (*i.e.* quantum chemistry).
  Their use for the Fermi-Hubbard model may not be appropriate!

.. autofunction:: transform_vec_to_skewmatrix

.. autofunction:: transform_vec_to_skewmatrix_with_active_space

.. autofunction:: prepare_vector_k_orbital_rotation_with_active_space

.. autofunction:: energy_cost_function_orbital_optimization

.. autofunction:: brute_force_orbital_optimization

.. autofunction:: sa_build_mo_hessian_and_gradient

.. autofunction:: build_mo_gradient

.. autofunction:: orbital_optimisation_newtonraphson

.. autofunction:: orbital_optimisation_newtonraphson_no_active_space

.. autofunction:: build_generalized_fock_matrix

.. autofunction:: build_generalized_fock_matrix_active_space_adapted



|
|
|

_______________

_______________



Bosonic systems
===============

.. currentmodule:: quantnbody.bosonic.tools

In this subsection, we list a series of functions useful for bosonic systems.


Basic functions for the creation of many-boson systems
----------------------------------------------------------------

.. autofunction:: build_nbody_basis

.. autofunction:: build_operator_a_dagger_a

_______________

_______________


Many-body Hamiltonians and excitations operators
--------------------------------------------------------

.. autofunction:: build_hamiltonian_bose_hubbard


_______________

_______________

Creating/manipulating/visualizing many-body wavefunctions
------------------------------------------------------------------------

.. autofunction:: my_state

.. autofunction:: visualize_wft

_______________

_______________

Reduced density matrices
--------------------------------

.. autofunction:: build_1rdm

.. autofunction:: build_2rdm




_______________

_______________


Functions to manipulate bosonic integrals
------------------------------------------------

.. autofunction:: transform_1_2_body_tensors_in_new_basis

_______________

_______________
