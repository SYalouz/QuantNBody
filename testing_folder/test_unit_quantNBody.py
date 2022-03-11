import random
import unittest
TEST_CPP_ACCELERATED = False
if TEST_CPP_ACCELERATED:
    import pybind.Quant_NBody_fast as Quant_NBody
else:
    import Quant_NBody
import testing_folder.Quant_NBody_main_branch as Quant_NBody_old # This is the original library that I compare with.
import numpy as np
import parameterized  # conda install -c conda-forge parameterized
from tqdm import tqdm
import math
import pyscf
from pyscf import gto, scf, ao2mo, mcscf, fci
import psi4


# list_of_pubchem_ids = [962] --> water
def generate_atom_list_from_pubchem(pubchem_id):
    # Get molecules coordinates
    mol = psi4.geometry(f"pubchem:{pubchem_id}")
    mol.update_geometry()
    xyz_string = mol.save_string_xyz()

    # Generate atom list that is compatible with gto
    atom_list = []
    for line in xyz_string.split('\n')[1:-1]:
        line = line.split(' ')
        line = [i for i in line if i != '']
        print(line)
        atom_list.append([line[0], (float(line[1]), float(line[2]), float(line[3]))])
    print(atom_list)

    return atom_list


def generate_molecule(atom_list):
    # generate gto molecule object
    mol = gto.Mole()
    mol.build(atom=atom_list,  # in Angstrom
              basis='STO-3G',
              symmetry=False,
              spin=0)
    N_elec = mol.nelectron
    N_MO = mol.nao_nr()

    # MO coefficients
    mf = scf.RHF(mol)
    mf.kernel()

    # generation of 1 and 2 electron integrals
    h_MO = np.einsum('pi,pq,qj->ij', mf.mo_coeff,
                     mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc'),
                     mf.mo_coeff)
    g_MO = ao2mo.kernel(mol, mf.mo_coeff)
    g_MO = ao2mo.restore('s1', g_MO, N_MO)

    E_rep_nuc = mol.energy_nuc()  # Nuclei repuslion energy
    return h_MO, g_MO, N_elec, N_MO, E_rep_nuc


def generate_random_xyz_he_h2(seed=1):
    random.seed(seed)
    d1 = 0.3 * (random.random() - 0.5) + 1.333
    d2 = 0.3 * (random.random() - 0.5) + 1.333
    phi2 = 1.2 * (random.random() - 0.5)
    print(f'phi = {phi2 * 180 / 3.14}')
    XYZ_geometry = f""" He   0.   0.  0.
 H   -{d1}  0.  0. 
 H  {d2 * np.cos(phi2)}   {d2 * np.sin(phi2)}  0."""
    return XYZ_geometry


def generate_from_graph(sites, connections):
    """
    We can provide graph information and program generates hamiltonian automatically
    :param sites: in the type of: {0:{'v':0, 'U':4}, 1:{'v':1, 'U':4}, 2:{'v':0, 'U':4}, 3:{'v':1, 'U':4}}
    :param connections: {(0, 1):1, (1, 2):1, (2, 3):1, (0,3):1}
    :return: h and U parameters
    """
    n_sites = len(sites)
    t = np.zeros((n_sites, n_sites), dtype=np.float64)
    v = np.zeros(n_sites, dtype=np.float64)
    u = np.zeros(n_sites, dtype=np.float64)
    for id, params in sites.items():
        if 'U' in params:
            u[id] = params['U']
        elif 'u' in params:
            u[id] = params['u']
        else:
            raise "Problem with params: " + params
        v[id] = params['v']
    for pair, param in connections.items():
        t[pair[0], pair[1]] = -param
        t[pair[1], pair[0]] = -param
    return t, v, u


# Parameters setting:
PARAM_TEST_FIRST = []
for n_mo1 in range(1, 8):
    for n_electrons1 in range(1, min(9, n_mo1 * 2)):
        PARAM_TEST_FIRST.append([f'N-MO={n_mo1}_N-e={n_electrons1}', n_mo1, n_electrons1])

PARAM_TEST_TRANSFORMATIONS = [
    ('0_10_2_F_F', 0, 10, 2, False, False),
    ('1_10_2_F_F', 1, 10, 2, False, False),
    ('2_10_2_F_F', 2, 10, 2, False, False),
    ('3_20_2_F_F', 3, 20, 2, False, False),
    ('4_40_2_F_F', 4, 40, 4, False, False),
]


class TestQuantNBody(unittest.TestCase):
    @parameterized.parameterized.expand(PARAM_TEST_FIRST)
    def test_first(self, name, n_mo, n_electrons):
        # Check if build_a_dagger_a and build_nbody_basis work based on different number of orbitals and electrons
        print(name)
        with self.subTest(i=f"{n_mo}, {n_electrons}"):
            print(f'N_mo: {n_mo}, N_electrons: {n_electrons}')
            nbody_basis_new = Quant_NBody.build_nbody_basis(n_mo, n_electrons)
            nbody_basis_old = Quant_NBody_old.Build_NBody_Basis(n_mo, n_electrons)
            self.assertEqual(nbody_basis_new.tolist(), nbody_basis_old)

            a_dagger_a_new = Quant_NBody.build_operator_a_dagger_a(nbody_basis_new)
            a_dagger_a_old = Quant_NBody_old.Build_operator_a_dagger_a(nbody_basis_old)
            shape1 = a_dagger_a_new.shape
            shape2 = a_dagger_a_old.shape
            print(shape1)
            self.assertEqual(shape1, shape2)
            for i in range(shape1[0]):
                for j in range(shape1[1]):
                    self.assertTrue(np.allclose(a_dagger_a_new[i, j].A, a_dagger_a_old[i, j].A))


    def test_build_hamiltonian(self):
        # I assume that a_dagger_a and nbody_basis are same from the previous test.
        # I generated imaginary molecule H - He - H with variable lengths and angle. I did this to be sure that I don't
        # have any symmetry.
        n_mo = 3
        n_electron = 4
        nbody_basis = Quant_NBody.build_nbody_basis(n_mo, n_electron)
        a_dagger_a = Quant_NBody.build_operator_a_dagger_a(nbody_basis)

        for seed in np.arange(5, 20, 4):
            heh2 = generate_random_xyz_he_h2(seed)
            h_MO, g_MO, N_elec, N_MO, E_rep_nuc = generate_molecule(heh2)
            if n_mo != N_MO or n_electron != N_elec:
                raise Exception('Problem112233')

            H_new = Quant_NBody.build_hamiltonian_quantum_chemistry(h_MO, g_MO, nbody_basis, a_dagger_a)
            H_old = Quant_NBody_old.Build_Hamiltonian_Quantum_Chemistry(h_MO, g_MO, nbody_basis, a_dagger_a)

            self.assertTrue(np.allclose(H_new.A, H_old.A))

    @parameterized.parameterized.expand([
        ["0_5_5", 0, 5, 5],
        ["0_6_5", 0, 6, 5],
        ["0_5_10", 0, 5, 10],
    ])
    def test_build_fh_hamiltonian(self, name, seed=0, site_number=5, bond_number=5):
        """
        Here I generate random molecules with random parameters.
        Parameters
        ----------
        name: used for parameterized library
        seed: Seed for random generator
        site_number: Number of sites in the molecule
        bond_number: Number of bonds (actual number could be smaller)

        Returns
        -------

        """
        print(name)
        n_mo = site_number
        n_electron = site_number
        nbody_basis = Quant_NBody.build_nbody_basis(n_mo, n_electron)
        a_dagger_a = Quant_NBody.build_operator_a_dagger_a(nbody_basis)

        for i in range(5):
            """Checks build of Fermi Hubbard Hamiltonian and corresponding 2rdm"""
            random.seed(seed)
            sites = {}
            edges = {}
            for site in range(site_number):
                sites[site] = {'U': random.random() * 10, 'v': (random.random() - 0.5) * 3}

            for bond in range(bond_number):
                f = random.randrange(0, site_number)
                t = random.randrange(0, site_number)
                if f != t:
                    edges[(f, t)] = random.random() * 2
            print(sites, edges)
            t, v, u1d = generate_from_graph(sites, edges)
            u = np.zeros((site_number, site_number, site_number, site_number))
            u[np.diag_indices(site_number, ndim=4)] = u1d

            H_new = Quant_NBody.build_hamiltonian_fermi_hubbard(t + v, u, nbody_basis, a_dagger_a)
            H_old = Quant_NBody_old.Build_Hamiltonian_Fermi_Hubbard(t + v, u, nbody_basis, a_dagger_a)

            self.assertTrue(np.allclose(H_new.A, H_old.A))

            eig_energies_new, eig_vectors_new = np.linalg.eigh(H_new.A)
            WFT_new = eig_vectors_new[:, 0]

            eig_energies_old, eig_vectors_old = np.linalg.eigh(H_old.A)
            WFT_old = eig_vectors_old[:, 0]

            self.assertTrue(np.allclose(Quant_NBody.build_2rdm_fh(WFT_new, a_dagger_a),
                                        Quant_NBody_old.Build_two_RDM_FH(WFT_old, a_dagger_a)))

    def test_rdms(self):
        # I will generate molecule of water in STO-3G basis set and compare all generated 1RDMS
        atom_list = generate_atom_list_from_pubchem(962)
        h_MO, g_MO, N_elec, N_MO, E_rep_nuc = generate_molecule(atom_list)

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

    @parameterized.parameterized.expand(PARAM_TEST_TRANSFORMATIONS)
    def test_transformations(self, name, seed, size=10, size_cluster=2, ref_result_hh=False, ref_result_block_hh=False):
        print(name)
        np.random.seed(seed)
        matrix1 = np.random.random((size, size))
        matrix1 = matrix1 + matrix1.T

        mat_hh_new = Quant_NBody.householder_transformation(matrix1)
        mat_hh_old = Quant_NBody_old.Householder_transformation(matrix1)
        self.assertTrue(np.allclose(mat_hh_new[0], mat_hh_old[0]))
        self.assertTrue(np.allclose(mat_hh_new[1], mat_hh_old[1]))
        if ref_result_hh:
            # maybe reference should be the function and not matrix!
            self.assertTrue(np.allclose(mat_hh_new[0], ref_result_hh))

        mat_block_hh_new = Quant_NBody.block_householder_transformation(matrix1, size_cluster)
        mat_block_hh_old = Quant_NBody_old.Block_householder_transformation(matrix1, size_cluster)
        self.assertTrue(np.allclose(mat_block_hh_new[0], mat_block_hh_old[0]))
        self.assertTrue(np.allclose(mat_block_hh_new[1], mat_block_hh_old[1]))
        if ref_result_block_hh:
            self.assertTrue(np.allclose(mat_block_hh_new[0], ref_result_block_hh))


# List of all function used in the package
# build_nbody_basis                         test_first
# check_sz                                  UNTESTED
# build_operator_a_dagger_a                 test_first
# build_mapping                             nested in test_first
# make_integer_out_of_bit_vector            nested in test_first
# new_state_after_sq_fermi_op               nested in test_first
# build_final_state_ad_a                    nested in test_first
# my_state                                  UNTESTED
# build_hamiltonian_quantum_chemistry       test_build_hamiltonian
# build_hamiltonian_fermi_hubbard           test_build_fh_hamiltonian
# fh_get_active_space_integrals             UNTESTED
# qc_get_active_space_integrals             UNTESTED
# build_s2_sz_splus_operator                UNTESTED
# build_1rdm_alpha                          test_rdms
# build_1rdm_beta                           test_rdms
# build_1rdm_spin_free                      test_rdms
# build_2rdm_fh                             test_build_fh_hamiltonian
# build_2rdm_spin_free                      test_rdms
# build_1rdm_and_2rdm_spin_free             test_rdms
# visualize_wft                             UNTESTED
# transform_1_2_body_tensors_in_new_basis   UNTESTED
# householder_transformation                test_transformations
# block_householder_transformation          test_transformations
# build_mo_1rdm_and_2rdm                    UNTESTED
# generate_h_chain_geometry                 UNTESTED
# generate_h_ring_geometry                  UNTESTED
# delta                                     UNTESTED


if __name__ == "__main__":
    unittest.main()
