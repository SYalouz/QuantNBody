import unittest
import Quant_NBody
import testing_folder.Quant_NBody_main_branch as Quant_NBody_old
import numpy as np
import e_operator

class TestQuantNBody(unittest.TestCase):

    def test_first(self):
        # Check if build_a_dagger_a and build_nbody_basis work based on different number of orbitals and electrons
        for n_mo in range(1, 8):
            for n_electrons in range(1, min(9, n_mo * 2)):
                print(f'N_mo: {n_mo}, N_electrons: {n_electrons}')
                nbody_basis_new = Quant_NBody.build_nbody_basis(n_mo, n_electrons)
                nbody_basis_old = Quant_NBody_old.Build_NBody_Basis(n_mo, n_electrons)
                self.assertEqual(nbody_basis_new, nbody_basis_old)

                a_dagger_a_new = Quant_NBody.build_operator_a_dagger_a(nbody_basis_new)
                a_dagger_a_old = Quant_NBody_old.Build_operator_a_dagger_a(nbody_basis_old)
                shape1 = a_dagger_a_new.shape
                shape2 = a_dagger_a_old.shape
                print(shape1)
                self.assertEqual(shape1, shape2)
                for i in range(shape1[0]):
                    for j in range(shape1[1]):
                        self.assertTrue(np.allclose(a_dagger_a_new[i, j].A, a_dagger_a_old[i, j].A))

    # check_sz, my_state weren't checked

    def test_build_hamiltonian(self):
        # I assume that a_dagger_a and nbody_basis are same from the previous test
        n_mo = 3
        n_electron = 4
        nbody_basis = Quant_NBody.build_nbody_basis(n_mo, n_electron)
        a_dagger_a = Quant_NBody.build_operator_a_dagger_a(nbody_basis)

        for seed in np.arange(5, 20, 4):
            heh2 = e_operator.generate_random_xyz_he_h2(seed)
            h_MO, g_MO, N_elec, N_MO, E_rep_nuc = e_operator.generate_molecule(heh2)
            if n_mo != N_MO or n_electron != N_elec:
                raise Exception( 'Problem112233')

            H_new = Quant_NBody.build_hamiltonian_quantum_chemistry(h_MO, g_MO, nbody_basis, a_dagger_a)
            H_old = Quant_NBody_old.Build_Hamiltonian_Quantum_Chemistry(h_MO, g_MO, nbody_basis, a_dagger_a)

            self.assertTrue(np.allclose(H_new.A, H_old.A))

    def test_rdms(self):
        # I will generate molecule of water in STO-3G basis set and compare all generated 1RDMS
        atom_list = e_operator.generate_atom_list_from_pubchem(962)
        h_MO, g_MO, N_elec, N_MO, E_rep_nuc = e_operator.generate_molecule(atom_list)

        nbody_basis = Quant_NBody.build_nbody_basis(N_MO, N_elec)
        a_dagger_a = Quant_NBody.build_operator_a_dagger_a(nbody_basis)
        H = Quant_NBody.build_hamiltonian_quantum_chemistry(h_MO, g_MO, nbody_basis, a_dagger_a)
        eig_energies, eig_vectors = np.linalg.eigh(H.A)
        WFT = eig_vectors[:, 0]

        self.assertTrue(np.allclose(Quant_NBody.build_1rdm_beta(WFT, a_dagger_a),
                         Quant_NBody_old.Build_One_RDM_beta(WFT, a_dagger_a)))

        self.assertTrue(np.allclose(Quant_NBody.build_1rdm_alpha(WFT, a_dagger_a),
                         Quant_NBody_old.Build_One_RDM_alpha(WFT, a_dagger_a)))


        a = Quant_NBody.build_1rdm_spin_free(WFT, a_dagger_a)
        b = Quant_NBody_old.Build_One_RDM_spin_free(WFT, a_dagger_a)

        self.assertTrue(np.allclose(Quant_NBody.build_1rdm_spin_free(WFT, a_dagger_a),
                         Quant_NBody_old.Build_One_RDM_spin_free(WFT, a_dagger_a)))

        self.assertTrue(np.allclose(Quant_NBody.build_2rdm_spin_free(WFT, a_dagger_a),
                         Quant_NBody_old.Build_two_RDM_spin_free(WFT, a_dagger_a)))

        ret_new = Quant_NBody.build_1rdm_and_2rdm_spin_free(WFT, a_dagger_a)
        ret_old = Quant_NBody_old.Build_One_and_Two_RDM_spin_free(WFT, a_dagger_a)
        self.assertTrue(np.allclose(ret_new[0], ret_old[0]))
        self.assertTrue(np.allclose(ret_new[1], ret_old[1]))


if __name__ == "__main__":
    unittest.main()
