import unittest
import quantnbody as qnb
import scipy
import psi4

BASIS_SET = "sto-3g"
GEOMETRY_H2 = "H   0.   0.   0.\nH   0.74   0.   0."
GEOMETRY_H_BE_H = "H   -1.33   0.   0.\nBe   0.   0.   0.\nH   1.33   0.   0."

def calculate_reference_results():
    # calculation of reference result
    psi4.core.clean()
    psi4.core.clean_variables()
    psi4.core.clean_options()
    psi4.core.set_output_file("output_psi4.txt", False)
    psi4.geometry(GEOMETRY_H2 + "\nsymmetry c1")
    psi4.set_options({'basis': BASIS_SET,
                      'num_roots': 2})
    fci = psi4.energy('FCI', return_wfn=False)
    E0_fci = psi4.variable('CI ROOT 0 TOTAL ENERGY')

    print(E0_fci)
    with open('h2_fci.dat', "w") as connection:
        connection.write(str(E0_fci))

    #  HBeH
    psi4.core.clean()
    psi4.core.clean_variables()
    psi4.core.clean_options()
    psi4.core.set_output_file("output_psi4.txt", False)
    psi4.geometry(GEOMETRY_H_BE_H + "\nsymmetry c1")
    psi4.set_options({'basis': BASIS_SET,
                      'num_roots': 2})
    fci = psi4.energy('FCI', return_wfn=False)
    E0_fci = psi4.variable('CI ROOT 0 TOTAL ENERGY')
    print(E0_fci)
    with open('h_be_h_fci.dat', "w") as connection:
        print(123123123213)
        connection.write(str(E0_fci))



class MyTestCase(unittest.TestCase):
    def test_h2_fci(self):
        nbody_basis = qnb.fermionic.tools.build_nbody_basis(2, 2)
        a_dagger_a = qnb.fermionic.tools.build_operator_a_dagger_a(nbody_basis)

        overlap_AO, h_AO, g_AO, C_ref, E_HF, E_rep_nuc = qnb.fermionic.tools.get_info_from_psi4(GEOMETRY_H2, BASIS_SET, 0)
        h_MO, g_MO = qnb.fermionic.tools.transform_1_2_body_tensors_in_new_basis(h_AO, g_AO, C_ref)
        H = qnb.fermionic.tools.build_hamiltonian_quantum_chemistry(h_MO, g_MO, nbody_basis, a_dagger_a)
        eig_en, eig_vec = scipy.linalg.eigh(H.A)
        E_0 = eig_en[0] + E_rep_nuc

        with open("h2_fci.dat", 'r') as conn:
            reference_result = float(conn.read().strip())
        print(reference_result, E_0)
        self.assertAlmostEqual(E_0, reference_result, 12)

    def test_h_be_h_fci(self):
        nbody_basis = qnb.fermionic.tools.build_nbody_basis(7, 6)
        a_dagger_a = qnb.fermionic.tools.build_operator_a_dagger_a(nbody_basis)
        overlap_AO, h_AO, g_AO, C_ref, E_HF, E_rep_nuc = qnb.fermionic.tools.get_info_from_psi4(GEOMETRY_H_BE_H,
                                                                                                BASIS_SET, 0)
        h_MO, g_MO = qnb.fermionic.tools.transform_1_2_body_tensors_in_new_basis(h_AO, g_AO, C_ref)
        print(h_MO.shape, g_MO.shape)
        H = qnb.fermionic.tools.build_hamiltonian_quantum_chemistry(h_MO, g_MO, nbody_basis, a_dagger_a)
        eig_en, eig_vec = scipy.linalg.eigh(H.A)
        E_0 = eig_en[0] + E_rep_nuc

        with open("h_be_h_fci.dat", 'r') as conn:
            reference_result = float(conn.read().strip())
        print(reference_result, E_0)
        self.assertAlmostEqual(E_0, reference_result, 8)

    def test_h2_cas(self):
        self.assertEqual(True, False)

    def test_h_be_h_cas(self):
        self.assertEqual(True, False)

    def test_fermi_hubbard(self):
        self.assertEqual(True, False)

    def test_bose_hubbard(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    # calculate_reference_results()
    unittest.main()
