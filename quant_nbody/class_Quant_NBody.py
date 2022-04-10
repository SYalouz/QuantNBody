import Quant_NBody
import numpy as np
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


class QuantNBody:
    def __init__(self, N_MO, N_electron, S_z_cleaning=False, override_NBody_basis=tuple()):
        self.n_mo = N_MO
        self.n_electron = N_electron
        if not override_NBody_basis:
            self.nbody_basis = Quant_NBody.build_nbody_basis(N_MO, N_elec, S_z_cleaning)
        else:
            self.nbody_basis = override_NBody_basis
        self.a_dagger_a = []

        self.H = scipy.sparse.csr_matrix((1, 1))
        self.h = np.array([], dtype=np.float64)
        self.U = np.array([], dtype=np.float64)
        self.g = np.array([], dtype=np.float64)
        self.ei_val = self.ei_vec = np.array([])
        self.WFT_0 = np.array([])
        self.one_rdm = np.array([])  # IT IS ONLY SPIN ALPHA!!!!!!
        self.two_rdm = np.array([])

    def build_operator_a_dagger_a(self, silent=False):
        self.a_dagger_a = Quant_NBody.build_operator_a_dagger_a(self.nbody_basis, silent)

    def build_hamiltonian_quantum_chemistry(self, h_, g_, S_2=None, S_2_target=None, penalty=100):
        """
            Create a matrix representation of the electronic structure Hamiltonian in any
            extended many-body basis

            Parameters
            ----------
            h_          :  One-body integrals
            g_          :  Two-body integrals
            S_2         :  Matrix representation of the S_2 operator (default is None)
            S_2_target  :  Value of the S_2 mean value we want to target (default is None)
            penalty     :  Value of the penalty term for state not respecting the spin symmetry (default is 100).

            Returns
            -------
            H_chemistry :  Matrix representation of the electronic structure Hamiltonian

        """
        self.h = h_
        self.g = g_
        self.H = Quant_NBody.build_hamiltonian_quantum_chemistry(h_, g_, self.nbody_basis, self.a_dagger_a, S_2,
                                                                 S_2_target, penalty)

    def build_hamiltonian_fermi_hubbard(self, h_, U_, S_2=None, S_2_target=None, penalty=100, v_term=None):
        """
            Create a matrix representation of the Fermi-Hubbard Hamiltonian in any
            extended many-body basis.

            Parameters
            ----------
            h_          :  One-body integrals
            U_          :  Two-body integrals (u[i,j,k,l] corresponds to a^+_i↑ a_j↑ a^+_k↓ a_l↓)
            S_2         :  Matrix representation of the S_2 operator (default is None)
            S_2_target  :  Value of the S_2 mean value we want to target (default is None)
            penalty     :  Value of the penalty term for state not respecting the spin symmetry (default is 100).
            v_term      :  4D matrix that is already transformed into correct representation.
                           (v_term[i,j,k,l] corresponds to E_ij E_kl)

            Returns
            -------
            H_fermi_hubbard :  Matrix representation of the Fermi-Hubbard Hamiltonian

            """
        self.h = h_
        self.U = U_
        self.H = Quant_NBody.build_hamiltonian_fermi_hubbard(h_, U_, self.nbody_basis, self.a_dagger_a, S_2, S_2_target,
                                                             penalty, v_term)

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

    def diagonalize_hamiltonian(self, full: bool = False, number_of_states: int = 3) -> None:
        """
        This is a method that calculates Hamiltonian expectation values and wave functions from the generated
        Hamiltonian. By default we don't calculate all eigenvectors because we don't need more than ground state
        and in this way we can save some calculation time.
        Parameters
        ----------
        full              : boolean that decides if we do costly costly full search or only diagonalizing first
                            number_of_states states.
        number_of_states  : Number of states that are calculated if full == False

        Returns
        -------
        None
        """
        if len(self.H.A) == 0:
            print('You have to generate H first')
        if full:
            self.ei_val, self.ei_vec = np.linalg.eigh(self.H.A)
        else:
            self.ei_val, self.ei_vec = scipy.sparse.linalg.eigsh(self.H, k=number_of_states, which='SA')

        self.WFT_0 = self.ei_vec[:, 0]

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
        self.one_rdm = Quant_NBody.build_1rdm_alpha(self.ei_vec[:, index], self.a_dagger_a)
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
        self.one_rdm = Quant_NBody.build_1rdm_beta(self.ei_vec[:, index], self.a_dagger_a)
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
        self.one_rdm = Quant_NBody.build_1rdm_spin_free(self.ei_vec[:, index], self.a_dagger_a)
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

    # def calculate_2rdm_fh_with_v(self, v_tilde=None):
    #     """
    #     This generates an 2RDM that we need to calculate on-site repulsion and optionally v-term.
    #     :return:
    #     """
    #     n_mo = self.n_mo
    #     two_rdm = np.zeros((n_mo, n_mo, n_mo, n_mo))
    #     two_rdm[:] = np.nan
    #     big_e_ = np.empty((2 * n_mo, 2 * n_mo), dtype=object)
    #     for p in range(n_mo):
    #         for q in range(n_mo):
    #             big_e_[p, q] = self.a_dagger_a[2 * p, 2 * q] + self.a_dagger_a[2 * p + 1, 2 * q + 1]
    #
    #     for p in range(n_mo):
    #         for q in range(n_mo):
    #             for r in range(n_mo):
    #                 for s in range(n_mo):
    #                     if v_tilde is not None and v_tilde[p, q, r, s] != 0:
    #                         two_rdm[p, q, r, s] = self.WFT_0.T @ big_e_[p, q] @ big_e_[r, s] @ self.WFT_0
    #                     elif p == q == r == s:
    #                         two_rdm[p, q, r, s] = self.WFT_0.T @ self.a_dagger_a[2 * p, 2 * q] @ \
    #                                               self.a_dagger_a[2 * r + 1, 2 * s + 1] @ self.WFT_0
    #     two_rdm[np.isnan(two_rdm)] = 0
    #     self.two_rdm = two_rdm
    #     return two_rdm

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
        self.one_rdm, self.two_rdm = Quant_NBody.build_1rdm_and_2rdm_spin_free(self.ei_vec[:, index], self.a_dagger_a)

    def visualize_wft(self, index=0, wft: Typing.Union[bool, List] = False, cutoff=0.005, atomic_orbitals=False):
        """
        Print the decomposition of a given input wave function in a many-body basis.

       Parameters
        ----------
        index            : If wft is False then it plots calculated wave functions (self.ei_vec) with given index.
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
            wf = self.ei_vec[:, index]
        else:
            if len(self.nbody_basis != wft):
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
