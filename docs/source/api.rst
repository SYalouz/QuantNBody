Fermionic systems
=============

.. currentmodule:: quantnbody.fermionic.tools

In this subsection, there are written functions for fermionic particles.


_______________

Basic functions for the creation of many-fermion operators
----------------------------------------------------------------
.. autofunction:: build_nbody_basis
_______________

.. autofunction:: build_operator_a_dagger_a
_______________

.. autofunction:: check_sz
_______________

.. autofunction:: build_mapping
_______________

.. autofunction:: make_integer_out_of_bit_vector
_______________

.. autofunction:: new_state_after_sq_fermi_op
_______________

.. autofunction:: build_final_state_ad_a
_______________




Many-body Hamiltonians and excitations operators
--------------------------------------------------------
.. autofunction:: build_hamiltonian_quantum_chemistry
_______________

.. autofunction:: build_hamiltonian_fermi_hubbard
_______________

.. autofunction:: build_E_and_e_operators
_______________
_______________

Spin operators
--------
.. autofunction:: build_s2_sz_splus_operator
_______________

.. autofunction:: build_s2_local
_______________

.. autofunction:: build_sAsB_coupling
_______________

.. autofunction:: build_spin_subspaces
_______________
_______________


Reduced density matrices
--------------------------------
.. autofunction:: build_full_mo_1rdm_and_2rdm_for_AS
_______________

.. autofunction:: build_1rdm_alpha
_______________

.. autofunction:: build_1rdm_beta
_______________

.. autofunction:: build_1rdm_spin_free
_______________

.. autofunction:: build_2rdm_fh_on_site_repulsion
_______________

.. autofunction:: build_2rdm_fh_dipolar_interactions
_______________

.. autofunction:: build_2rdm_spin_free
_______________

.. autofunction:: build_1rdm_and_2rdm_spin_free
_______________

.. autofunction:: build_hybrid_1rdm_alpha_beta
_______________

.. autofunction:: build_transition_1rdm_alpha
_______________

.. autofunction:: build_transition_1rdm_beta
_______________

.. autofunction:: build_transition_1rdm_spin_free
_______________

.. autofunction:: build_transition_2rdm_spin_free
_______________
_______________

Functions to create/manipulate/visualize many-body wavefunctions
------------------------------------------------------------------------
.. autofunction:: my_state
_______________

.. autofunction:: build_projector_active_space
_______________

.. autofunction:: build_penalty_orbital_occupancy
_______________

.. autofunction:: weight_det
_______________

.. autofunction:: scalar_product_different_MO_basis
_______________

.. autofunction:: transform_psi_MO_basis1_in_MO_basis2
_______________

.. autofunction:: TEST_transform_psi_MO_basis1_in_MO_basis2
_______________

.. autofunction:: scalar_product_different_MO_basis_with_frozen_orbitals
_______________

.. autofunction:: visualize_wft

_______________
_______________

Functions to manipulate fermionic integrals
------------------------------------------------

.. autofunction:: transform_1_2_body_tensors_in_new_basis
_______________

.. autofunction:: fh_get_active_space_integrals
_______________

.. autofunction:: fh_get_active_space_integrals_with_V
_______________

.. autofunction:: qc_get_active_space_integrals

_______________
_______________

Quantum embedding transformations (Householder)
--------------------------------------------------------

.. autofunction:: householder_transformation
_______________

.. autofunction:: block_householder_transformation

_______________
_______________

Psi4 calculation helper
------------------------

.. autofunction:: get_info_from_psi4

_______________

.. autofunction:: generate_h_chain_geometry

_______________

.. autofunction:: generate_h_ring_geometry

_______________

.. autofunction:: generate_h4_geometry

_______________
_______________

Miscellaneous
----------------

.. autofunction:: get_ket_in_atomic_orbitals

.. autofunction:: compute_energy_with_rdm

.. autofunction:: prepare_vector_k_orbital_rotation_fwith_active_space

.. autofunction:: transform_vec_to_skewmatrix_with_active_space

.. autofunction:: energy_cost_function_orbital_optimization

.. autofunction:: brute_force_orbital_optimization

.. autofunction:: filter_h_g_orb

.. autofunction:: sa_build_mo_hessian_and_gradient

.. autofunction:: sa_build_mo_hessian_and_gradient_no_active_space

.. autofunction:: build_mo_gradient

.. autofunction:: orbital_optimisation_newtonraphson

.. autofunction:: orbital_optimisation_newtonraphson_no_active_space

.. autofunction:: transform_vec_to_skewmatrix

.. autofunction:: build_generalized_fock_matrix

.. autofunction:: build_generalized_fock_matrix_active_space_adapted

.. autofunction:: f_inactive

.. autofunction:: f_active

.. autofunction:: q_aux

.. autofunction:: get_super_index

.. autofunction:: delta




fermionic.Hamiltonian
_____________________

.. currentmodule:: quantnbody.fermionic.class_file

In this subsection, a class that repachage some of the functions from `fermionic.tools` into methods.

.. autoclass:: Hamiltonian
   :members:



Bosonic systems
=============

.. currentmodule:: quantnbody.bosonic.tools

In this subsection, there is documentation for functions for bosonic particles.

.. autofunction:: build_nbody_basis

.. autofunction:: build_mapping

.. autofunction:: build_final_state_ad_a

.. autofunction:: new_state_after_sq_boson_op

.. autofunction:: build_operator_a_dagger_a

.. autofunction:: make_integer_out_of_bit_vector

.. autofunction:: my_state

.. autofunction:: build_hamiltonian_bose_hubbard

.. autofunction:: build_1rdm

.. autofunction:: build_2rdm

.. autofunction:: transform_1_2_body_tensors_in_new_basis

.. autofunction:: visualize_wft

.. autofunction:: delta



Important data structures
_________________________

There are some data structures that are important for correct functioning of `QuantNBody`

`Nbody_basis`
^^^^^^^^^^^^^

`Nbody_basis` is a 2D NumPy array. It is a list of all possible configurations; each element (row) of this array is a
configuration (ket). This configuration is represented by a list of occupation numbers of each spin orbital.

TODO

* dimensionality of array
* generated by build_nbody_basis


`a_dagger_a`
^^^^^^^^^^^^^

TODO

`Hamiltonian`
^^^^^^^^^^^^^

TODO

`mapping_kappa`
^^^^^^^^^^^^^^^

TODO
