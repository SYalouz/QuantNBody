# This class is for Martin. I am going to move it away after I stop developing it
import quantnbody as qnb
import numpy as np
import scipy
from . import Quant_NBody
import scipy.sparse
import matplotlib.pyplot as plt
import scipy.optimize
import typing
from typing import List


def print_matrix(matrix, plot_heatmap='', ret=False, output_formatting_number="+15.10f", output_separator='  '):
    ret_string = ""
    for line in matrix:
        l1 = ['{num:{dec}}'.format(num=cell, dec=output_formatting_number) for cell in line]
        ret_string += f'{output_separator}'.join(l1) + "\n"
    if plot_heatmap:
        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()
        plt.title(plot_heatmap)
    if ret:
        return ret_string
    print(ret_string, end='')


class HamiltonianV2(qnb.Hamiltonian):
    def __init__(self, n_mo, n_electron, S_z_cleaning=False, override_NBody_basis=tuple()):
        super().__init__(n_mo, n_electron, S_z_cleaning, override_NBody_basis)
        self.one_rdm = np.array([])
        self.two_rdm = np.array([])

    def my_state(self, slater_determinant: List[int]) -> List[int]:
        """
        Translate a Slater determinant (occupation number list) into a many-body
        state referenced into a given Many-body basis.

        Parameters
        ----------
        slater_determinant  : occupation number list

        Returns
        -------
        state :  The slater determinant referenced in the many-body basis
        """
        return Quant_NBody.my_state(slater_determinant, self.nbody_basis)

    def build_s2_sz_splus_operator(self):
        """
        Create a matrix representation of the spin operators s_2, s_z and s_plus
        in the many-body basis.

        Returns
        -------
        s_2, s_plus, s_z :  matrix representation of the s_2, s_plus and s_z operators
                            in the many-body basis.
        """
        return Quant_NBody.build_s2_sz_splus_operator(self.a_dagger_a)

    def calculate_1rdm_alpha(self, index=0):
        """
        Create a spin-alpha 1 RDM out of a given wave function

        Parameters
        ----------
        index :  Wave function for which we want to build the 1-RDM

        Returns
        -------
        one_rdm_alpha : spin-alpha 1-RDM
        """
        self.one_rdm = Quant_NBody.build_1rdm_alpha(self.eig_vectors[:, index], self.a_dagger_a)
        return self.one_rdm

    def calculate_1rdm_beta(self, index=0):
        """
        Create a spin-beta 1 RDM out of a given wave function

        Parameters
        ----------
        index :  Wave function for which we want to build the 1-RDM. Ground state corresponds to default
                 argument 0.

        Returns
        -------
        one_rdm_beta : spin-beta 1-RDM
        """
        self.one_rdm = Quant_NBody.build_1rdm_beta(self.eig_vectors[:, index], self.a_dagger_a)
        return self.one_rdm

    def calculate_1rdm_spin_free(self, index=0):
        """
        Create a spin-free 1 RDM out of a given wave function

        Parameters
        ----------
        index  : Wave function for which we want to build the 1-RDM. Ground state corresponds to default
                 argument 0.

        Returns
        -------
        one_rdm : Spin-free 1-RDM

        """
        self.one_rdm = Quant_NBody.build_1rdm_spin_free(self.eig_vectors[:, index], self.a_dagger_a)
        return self.one_rdm

    def build_2rdm_fh_on_site_repulsion(self, mask=None):
        """
        Create a spin-free 2 RDM out of a given Fermi Hubbard wave function for the on-site repulsion operator.
        (u[i,j,k,l] corresponds to a^+_i↑ a_j↑ a^+_k↓ a_l↓)

        Parameters
        ----------
        mask  : 4D array is expected. Function is going to calculate only elements of 2rdm where mask is not 0.
                For default None the whole 2RDM is calculated.
                if we expect 2RDM to be very sparse (has only a few non-zero elements) then it is better to provide
                array that ensures that we won't calculate elements that are not going to be used in calculation of
                2-electron interactions.
        Returns
        -------
        two_rdm for the on-site-repulsion operator
        """
        self.two_rdm = Quant_NBody.build_2rdm_fh_on_site_repulsion(self.WFT_0, self.a_dagger_a, mask=mask)
        return two_rdm

    def build_2rdm_fh_dipolar_interactions(self, mask=None):
        """
        Create a spin-free 2 RDM out of a given Fermi Hubbard wave function for the dipolar interaction operator
        it corresponds to <psi|(a^+_i↑ a_j↑ + a^+_i↓ a_j↓)(a^+_k↑ a_l↑ + a^+_k↓ a_l↓)|psi>

        Parameters
        ----------
        mask  : 4D array is expected. Function is going to calculate only elements of 2rdm where mask is not 0.
                For default None the whole 2RDM is calculated.
                If we expect 2RDM to be very sparse (has only a few non-zero elements) then it is better to provide
                array that ensures that we won't calculate elements that are not going to be used in calculation of
                2-electron interactions.
        Returns
        -------
        One_RDM_alpha : Spin-free 1-RDM
        """
        self.two_rdm = build_2rdm_fh_dipolar_interactions(self.WFT_0, self.a_dagger_a, mask=mask)
        return self.two_rdm

    def build_2rdm_spin_free(self):
        """
        Create a spin-free 2 RDM out of a given wave function

        Parameters
        ----------

        Returns
        -------
        2RDM : Spin-free 2-RDM

        """
        self.two_rdm = Quant_NBody.build_2rdm_spin_free(self.WFT_0, self.a_dagger_a)
        return self.two_rdm

    def build_1rdm_and_2rdm_spin_free(self, index=0):
        """
        Create a spin-free 2 RDM out of a given wave function

        Parameters
        ----------
        index  :  Index of wave function for which we want to build the 1-RDM

        Returns
        -------
        tuple(1RDM, 2RDM)  : Spin-free 1-RDM and spin-free 2-RDM
        """
        self.one_rdm, self.two_rdm = Quant_NBody.build_1rdm_and_2rdm_spin_free(self.eig_vectors[:, index],
                                                                               self.a_dagger_a)

    def visualize_wft(self, index=0, wft: Typing.Union[bool, List] = False, cutoff=0.005, atomic_orbitals=False):
        """
        Print the decomposition of a given input wave function in a many-body basis.

       Parameters
        ----------
        index            : If wft is False then it plots calculated wave functions (self.eig_vectors) with given index.
                           If wft is an iterable variable then this argument is ignored.
        wft              : If kept False then precalculated wave function is calculated.
                           Otherwise put a list of coefficients.
        cutoff           : Cut off for the amplitudes retained (default is 0.005)
        atomic_orbitals  : Boolean; If True then instead of 0/1 for spin orbitals we get 0/alpha/beta/2 for atomic
                           orbitals

        Returns
        -------
        Same string that was printed to the terminal (the wave function)
        """
        if isinstance(wft, bool) and not wft:
            wf = self.eig_vectors[:, index]
        else:
            if len(self.nbody_basis != len(wft)):
                raise IndexError(f"Length of wft is not correct ({len(wft)} vs {len(self.nbody_basis)}) ")
            wf = wft
        return Quant_NBody.visualize_wft(wf, self.nbody_basis, cutoff, atomic_orbitals)

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
        if self.eig_vectors.shape[0] == 0:
            raise Exception('diagonalization of Hamiltonian didnt happen yet')
        self.calculate_1rdm()
        occupations = self.one_rdm.diagonal()
        model = scipy.optimize.root(cost_function_v_hxc, starting_approximation,
                                    args=(occupations, self.h, self.n_electron))
        if not model.success or np.sum(np.square(model.fun)) > solution_classification_tolerance:
            return False
        energy_fci = self.eig_values[0]
        v_hxc = model.x
        h_ks = self.h + np.diag(v_hxc)
        eig_values, eig_vectors = np.linalg.eigh(h_ks)
        energy2 = 0
        for i in range(self.n_electron):
            # i goes over 0 to n_electron - 1
            energy2 += eig_values[i // 2]
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
    eig_values, eig_vectors = np.linalg.eigh(h_ks)
    one_rdm = generate_1rdm(eig_vectors.shape[0], Ne, eig_vectors)
    n_ks = one_rdm.diagonal()
    return n_ks - correct_values
