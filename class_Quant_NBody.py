import Quant_NBody
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib import cm

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
        self.n_elec = N_elec
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

    def build_operator_a_dagger_a(self):
        self.a_dagger_a = Quant_NBody.build_operator_a_dagger_a(self.nbody_basis)

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

    def calculate_1rdm_tot(self, index=0):
        return Quant_NBody.Build_One_RDM_spin_free(self.ei_vec[:, index], self.a_dagger_a)


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
