import unittest
import numpy as np
import psi4
import scipy
import quantnbody as qnb


class MyTestCase(unittest.TestCase):
    def test_FERMION_BASICS(self):
        # ========================================================|
        # Parameters for the simulation
        nelec_active = 4  # Number of active electrons in the Active-Space
        active_indices = [i for i in range(0, 3)]
        n_mo_active = len(active_indices)

        # Building the Many-body basis
        nbody_basis = qnb.fermionic.tools.build_nbody_basis(len(active_indices), nelec_active)

        # Building the matrix representation of the adagger_a operator in the many-body basis
        a_dagger_a = qnb.fermionic.tools.build_operator_a_dagger_a(nbody_basis)

        self.assertTrue(np.allclose(nbody_basis, np.load('nbodybasis_fermion_test.npy'), atol=1e-08),
                        'Error in many-body basis generation for fermionic systems')
        test_a_dagger_a = np.load('a_dagger_a_fermion_test.npy', allow_pickle=True)

        for mode in range(2 * n_mo_active):
            for mode_ in range(2 * n_mo_active):
                self.assertTrue(np.allclose(a_dagger_a[mode, mode_].A, test_a_dagger_a[mode, mode_].A, atol=1e-08),
                                'Error in a_dagger_a generation for fermionic systems')

        print(nbody_basis, np.load('nbodybasis_fermion_test.npy'))

    def test_FERMION_MODEL_HAMILTONIAN(self):
        # ========================================================|
        # Parameters for the simulation
        nelec_active = 4  # Number of active electrons in the Active-Space
        frozen_indices = [i for i in range(0)]
        active_indices = [i for i in range(0, 3)]
        virtual_indices = [i for i in range(4, 4)]

        n_mo = len(frozen_indices) + len(active_indices) + len(virtual_indices)

        # Building the Many-body basis
        nbody_basis = qnb.fermionic.tools.build_nbody_basis(len(active_indices), nelec_active)

        # Building the matrix representation of the adagger_a operator in the many-body basis
        a_dagger_a = qnb.fermionic.tools.build_operator_a_dagger_a(nbody_basis)

        # Building the matrix representation of several interesting spin operators in the many-body basis
        S_2, s_z, s_plus = qnb.fermionic.tools.build_s2_sz_splus_operator(a_dagger_a)

        U = 6

        # Hopping terms
        h_MO = np.zeros((n_mo, n_mo))
        for site in range(n_mo):
            for site_ in range(site, n_mo - 1):
                h_MO[site, site_] = h_MO[site_, site] = -1
        h_MO[0, n_mo - 1] = h_MO[n_mo - 1, 0] = -1

        # MO energies
        for site in range(n_mo):
            h_MO[site, site] = - site

        U_MO = np.zeros((n_mo, n_mo, n_mo, n_mo))
        for site in range(n_mo):
            U_MO[site, site, site, site] = U

        # Preparing the active space integrals and the associated core contribution
        E_core, h_, U_ = qnb.fermionic.tools.fh_get_active_space_integrals(h_MO,
                                                                           U_MO,
                                                                           frozen_indices=frozen_indices,
                                                                           active_indices=active_indices)

        # Building the matrix representation of the Hamiltonian operators
        H = qnb.fermionic.tools.build_hamiltonian_fermi_hubbard(h_,
                                                                U_,
                                                                nbody_basis,
                                                                a_dagger_a,
                                                                S_2=S_2,
                                                                S_2_target=0,
                                                                penalty=100,
                                                                v_term=None)
        eig_en, eig_vec = scipy.linalg.eigh(H.A)
        self.assertTrue(np.allclose(H.A, np.load('ham_fermionic_model_test.npy', allow_pickle=True)),
                        'Error in Hamiltonian generation for fermionic model systems')
        self.assertAlmostEqual(eig_en[0], -0.18328337002617445, 8,
                               'Error in groundstate energy estimation for fermionic model systems')
        self.assertAlmostEqual(eig_en[1], 1.3628124641217596, 8,
                               'Error in first excited state energy estimation for fermionic model systems')

    def test_FERMION_AB_INITIO_HAMILTONIAN(self):

        psi4.core.set_output_file("output_Psi4.txt", False)
        basisset = 'sto-3g'
        nelec_active = 2  # Number of active electrons in the Active-Space
        frozen_indices = [i for i in range(1)]
        active_indices = [i for i in range(1, 3)]

        # Building the Many-body basis
        nbody_basis = qnb.fermionic.tools.build_nbody_basis(len(active_indices), nelec_active)

        # Building the matrix representation of the adagger_a operator in the many-body basis
        a_dagger_a = qnb.fermionic.tools.build_operator_a_dagger_a(nbody_basis)

        # Building the matrix representation of several interesting spin operators in the many-body basis
        S_2, S_p, S_Z = qnb.fermionic.tools.build_s2_sz_splus_operator(a_dagger_a)

        # ========================================================|
        # Molecular geometry / Quantum chemistry calculations
        # Li-H geometry
        string_geo = """Li 0 0 0
                        H  0 0 {}
                        symmetry c1 """.format(2)

        molecule = psi4.geometry(string_geo)
        psi4.set_options({'basis': basisset,
                          'reference': 'rhf',
                          'SCF_TYPE': 'DIRECT'})

        scf_e, scf_wfn = psi4.energy('HF', molecule=molecule, return_wfn=True, verbose=0)
        C_RHF = np.asarray(scf_wfn.Ca()).copy()  # MO coeff matrix from the initial RHF calculation
        mints = psi4.core.MintsHelper(scf_wfn.basisset())  # Get AOs integrals using MintsHelper
        Num_AO = np.shape(np.asarray(mints.ao_kinetic()))[0]
        E_rep_nuc = molecule.nuclear_repulsion_energy()

        # Construction of the first reference Hamiltonian / MO integrals
        C_ref = C_RHF  # Initial MO coeff matrix

        # Storing the 1/2-electron integrals in the original AO basis
        h_AO = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        g_AO = np.asarray(mints.ao_eri()).reshape((Num_AO, Num_AO, Num_AO, Num_AO))

        h_MO, g_MO = qnb.fermionic.tools.transform_1_2_body_tensors_in_new_basis(h_AO, g_AO, C_ref)

        E_core, h_, g_ = qnb.fermionic.tools.qc_get_active_space_integrals(h_MO,
                                                                           g_MO,
                                                                           frozen_indices=frozen_indices,
                                                                           active_indices=active_indices)
        # Building the matrix representation of the Hamiltonian operators
        H = qnb.fermionic.tools.build_hamiltonian_quantum_chemistry(h_,
                                                                    g_,
                                                                    nbody_basis,
                                                                    a_dagger_a,
                                                                    S_2=S_2,
                                                                    S_2_target=0,
                                                                    penalty=100)

        eig_en, eig_vec = scipy.linalg.eigh(H.A)
        self.assertTrue(np.allclose(H.A, np.load('ham_fermionic_ab_initio_test.npy', allow_pickle=True)),
                        'Error in Hamiltonian generation for fermionic ab initio systems')
        self.assertAlmostEqual(eig_en[0] + E_core + E_rep_nuc, -7.831533642199216, 8,
                               'Error in groundstate energy estimation for fermionic ab initio systems')
        self.assertAlmostEqual(eig_en[1] + E_core + E_rep_nuc, -7.704470589024512, 8,
                               'Error in first excited state energy estimation for fermionic ab initio systems')

    def test_BOSON_MODEL_BASICS(self):
        n_mode = 4
        n_boson = 10
        nbodybasis = qnb.bosonic.tools.build_nbody_basis(n_mode, n_boson)
        a_dagger_a = qnb.bosonic.tools.build_operator_a_dagger_a(nbodybasis)

        self.assertTrue(np.allclose(nbodybasis, np.load('nbodybasis_boson_test.npy')),
                        'Error in many-body basis generation')
        test_a_dagger_a = np.load('a_dagger_a_boson_test.npy', allow_pickle=True)
        for mode in range(n_mode):
            for mode_ in range(n_mode):
                self.assertTrue(np.allclose(a_dagger_a[mode, mode_].A,
                                   test_a_dagger_a[mode, mode_].A), 'Error in groundstate energy estimation')

    def test_BOSON_MODEL_HAMILTONIAN(self):
        n_mode = 4
        n_boson = 10
        U = 1

        nbodybasis = qnb.bosonic.tools.build_nbody_basis(n_mode, n_boson)
        a_dagger_a = qnb.bosonic.tools.build_operator_a_dagger_a(nbodybasis)
        h_ = np.zeros((n_mode, n_mode))
        for site in range(n_mode):
            for site_ in range(n_mode):
                if (site != site_):
                    h_[site, site_] = h_[site_, site] = -1

        U_ = np.zeros((n_mode, n_mode, n_mode, n_mode))
        for site in range(n_mode):
            U_[site, site, site, site] = - U / 2

        # Building the matrix representation of the Hamiltonian operators
        H = qnb.bosonic.tools.build_hamiltonian_bose_hubbard(h_,
                                                             U_,
                                                             nbodybasis,
                                                             a_dagger_a)
        eig_en, eig_vec = scipy.linalg.eigh(H.A)
        self.assertTrue(np.allclose(H.A, np.load('ham_boson_test.npy',allow_pickle=True)),
                        'Error in Hamiltonian generation for bosonic systems')
        self.assertAlmostEqual(eig_en[0], -49.276698175251404, 8, 'Error in groundstate energy estimation')
        self.assertAlmostEqual(eig_en[1], -49.27144736991592, 8, 'Error in first excited state energy estimation')


if __name__ == '__main__':
    # calculate_reference_results()
    unittest.main()
