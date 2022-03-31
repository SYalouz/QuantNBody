import Quant_NBody
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize

OUTPUT_FORMATTING_NUMBER = "+15.10f"
OUTPUT_SEPARATOR = "  "


def print_matrix(matrix, plot_heatmap='', ret=False):
    ret_string = ""
    for line in matrix:
        l1 = ['{num:{dec}}'.format(num=cell, dec=OUTPUT_FORMATTING_NUMBER) for cell in line]
        ret_string += f'{OUTPUT_SEPARATOR}'.join(l1) + "\n"
    if plot_heatmap:
        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()
        plt.title(plot_heatmap)
    if ret:
        return ret_string
    print(ret_string, end='')


class QuantNBody:
    def __init__(self, N_MO, N_elec, S_z_cleaning=False, override_NBody_basis=tuple()):
        self.n_mo = N_MO
        self.n_electron = N_elec
        if not override_NBody_basis:
            self.nbody_basis = Quant_NBody.build_nbody_basis(N_MO, N_elec, S_z_cleaning)
        else:
            self.nbody_basis = override_NBody_basis
        self.a_dagger_a = []

        self.H = scipy.sparse.csr_matrix((1, 1))
        self.h = np.array([], dtype=np.float64)
        self.U = np.array([], dtype=np.float64)
        self.ei_val = self.ei_vec = np.array([])
        self.WFT_0 = np.array([])
        self.one_rdm = np.array([])  # IT IS ONLY SPIN ALPHA!!!!!!
        self.two_rdm = np.array([])

    def build_operator_a_dagger_a(self, silent=False):
        self.a_dagger_a = Quant_NBody.build_operator_a_dagger_a(self.nbody_basis, silent)

    def build_hamiltonian_quantum_chemistry(self, h_, U_, *args, **kwargs):
        self.h = h_
        self.U = U_
        self.H = Quant_NBody.build_hamiltonian_quantum_chemistry(h_, U_, self.nbody_basis, self.a_dagger_a,
                                                                 *args, **kwargs)

    def build_hamiltonian_fermi_hubbard(self, h_, U_, *args, **kwargs):
        self.h = h_
        self.U = U_
        self.H = Quant_NBody.build_hamiltonian_fermi_hubbard(h_, U_, self.nbody_basis, self.a_dagger_a,
                                                             *args, **kwargs)

    def diagonalize_hamiltonian(self):
        if len(self.H.A) == 0:
            print('You have to generate H first')

        self.ei_val, self.ei_vec = np.linalg.eigh(self.H.A)
        self.WFT_0 = self.ei_vec[:, 0]

    def visualize_coefficients(self, index, cutoff=0.005):
        # Quant_NBody.Visualize_WFT(self.ei_vec[index], self.nbody_basis, cutoff=cutoff)
        WFT = self.ei_vec[:, index]
        list_index = np.where(abs(WFT) > cutoff)[0]

        states = []
        coefficients = []
        for index in list_index:
            coefficients += [WFT[index]]
            states += [self.nbody_basis[index]]

        list_sorted_index = np.flip(np.argsort(np.abs(coefficients)))

        print()
        print('\t ----------- ')
        print('\t Coeff.      N-body state')
        print('\t -------     -------------')
        for index in list_sorted_index[0:8]:
            sign_ = '+'
            if abs(coefficients[index]) != coefficients[index]: sign_ = '-'
            print('\t', sign_, '{:1.5f}'.format(abs(coefficients[index])),
                  '\t' + get_better_ket(states[index]))

    def calculate_1rdm(self, index=0):
        """THIS CALCULATES ONLY SPIN ALPHA!!"""
        self.one_rdm = Quant_NBody.build_1rdm_alpha(self.ei_vec[:, index], self.a_dagger_a)
        return self.one_rdm

    def calculate_1rdm_spin_free(self, index=0):
        """THIS CALCULATES ONLY SPIN ALPHA!!"""
        self.one_rdm = Quant_NBody.build_1rdm_spin_free(self.ei_vec[:, index], self.a_dagger_a)
        return self.one_rdm


    def calculate_2rdm_fh(self, index=0):
        """THIS CALCULATES ONLY SPIN ALPHA!!"""
        self.two_rdm = Quant_NBody.build_2rdm_fh(self.ei_vec[:, index], self.a_dagger_a)
        return self.two_rdm

    def calculate_v_hxc(self, starting_approximation, silent=False,
                        solution_classification_tolerance=1e-3) -> typing.Union[bool, np.array]:
        """
        This function calculates real Hxc potentials based on the real wave function. It does so by optimizing
        difference between real density and density obtained for KS system by some approximation for Hxc potentials.
        In the end it returns False if converged distances are too far one from another (sum square)
        Parameters
        ----------
        starting_approximation: First approximation for the Hxc potentials
        silent: If False then it prints resulting object from optimization algorithm
        solution_classification_tolerance: What is the highest acceptable sum square difference of KS density.

        Returns
        -------
        False if it didn't converge or array of Hxc potentials
        """
        if self.ei_vec.shape[0] == 0:
            raise Exception('diagonalization of Hamiltonian didnt happen yet')
        self.calculate_1rdm()
        occupations = self.one_rdm.diagonal()
        model = scipy.optimize.root(cost_function_v_hxc, starting_approximation,
                                    args=(occupations, self.h, self.n_electron))
        if not model.success or np.sum(np.square(model.fun)) > solution_classification_tolerance:
            return False
        energy_fci = self.ei_val[0]
        v_hxc = model.x
        h_ks = self.h + np.diag(v_hxc)
        ei_val, ei_vec = np.linalg.eigh(h_ks)
        one_rdm = generate_1rdm(self.n_mo, self.n_electron, ei_vec)
        energy2 = 0
        for i in range(self.n_electron):
            # i goes over 0 to N_electron - 1
            energy2 += ei_val[i // 2]
        v_hxc_2 = v_hxc + (energy_fci - energy2) / self.n_electron
        if not silent:
            print(model)
        return v_hxc_2

def generate_1rdm(Ns, Ne, wave_function):
    # generation of 1RDM
    if Ne % 2 != 0:
        raise f'problem with number of electrons!! Ne = {Ne}, Ns = {Ns}'

    y = np.zeros((Ns, Ns), dtype=np.float64)  # reset gamma
    for k in range(int(Ne / 2)):  # go through all orbitals that are occupied!
        vec_i = wave_function[:, k][np.newaxis]
        y += vec_i.T @ vec_i  #
    return y

def cost_function_v_hxc(v_hxc, correct_values, h, Ne):
    h_ks = h + np.diag(v_hxc)
    ei_val, ei_vec = np.linalg.eigh(h_ks)
    one_rdm = generate_1rdm(ei_vec.shape[0], Ne, ei_vec)
    n_ks = one_rdm.diagonal()
    return n_ks - correct_values


def get_better_ket(state, bra=False):
    ret_string = ""
    for i in range(len(state) // 2):
        if state[i * 2] == 1:
            if state[i * 2 + 1] == 1:
                ret_string += '2'
            else:
                ret_string += 'a'
        elif state[i * 2 + 1] == 1:
            ret_string += 'b'
        else:
            ret_string += '0'
    if bra:
        ret_string = '⟨' + ret_string + '|'
    else:
        ret_string = '|' + ret_string + '⟩'
    return ret_string
