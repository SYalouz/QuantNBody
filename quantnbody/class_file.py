from . import tools_file
import numpy as np
import scipy.sparse, scipy.sparse.linalg


def print_matrix(matrix, ret=False, output_formatting_number="+15.10f", output_separator='  '):
    ret_string = ""
    for line in matrix:
        l1 = ['{num:{dec}}'.format(num=cell, dec=output_formatting_number) for cell in line]
        ret_string += f'{output_separator}'.join(l1) + "\n"
    if ret:
        return ret_string
    print(ret_string, end='')


class Hamiltonian:
    def __init__(self, n_mo, n_electron, S_z_cleaning=False, override_NBody_basis=tuple()):
        self.n_mo = n_mo
        self.n_electron = n_electron
        if not override_NBody_basis:
            self.nbody_basis = tools_file.build_nbody_basis(n_mo, n_electron, S_z_cleaning)
        else:
            self.nbody_basis = override_NBody_basis
        self.a_dagger_a = []

        self.H = scipy.sparse.csr_matrix((1, 1))
        self.h = np.array([], dtype=np.float64)
        self.U = np.array([], dtype=np.float64)
        self.g = np.array([], dtype=np.float64)
        self.eig_values = self.eig_vectors = np.array([])
        self.E_ex = np.array([])
        self.e_ex = np.array([])

    def build_operator_a_dagger_a(self, silent=False):
        """
        Function that builds a_dagger_a matrix
        Parameters
        ----------
        silent  : If silent is True then message is printed to the console when build is finished

        Returns
        -------
        None, a_dagger_a gets saved to the object_name.a_dagger_a
        """
        self.a_dagger_a = tools_file.build_operator_a_dagger_a(self.nbody_basis, silent)

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
            Matrix representation of the electronic structure Hamiltonian gets saved to the object_name.H

        """
        self.h = h_
        self.g = g_
        self.H = tools_file.build_hamiltonian_quantum_chemistry(h_, g_, self.nbody_basis, self.a_dagger_a, S_2,
                                                                 S_2_target, penalty)
        self.E_ex = tools_file.E_
        self.e_ex = tools_file.e_

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
            Matrix representation of the Fermi-Hubbard Hamiltonian gets saved to the object_name.H

            """
        self.h = h_
        self.U = U_
        self.H = tools_file.build_hamiltonian_fermi_hubbard(h_, U_, self.nbody_basis, self.a_dagger_a, S_2,
                                                               S_2_target, penalty, v_term)
        self.E_ex = tools_file.E_
        self.e_ex = tools_file.e_

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
        Eigenvalues and eigenvectors get saved to the object_name.eig_values and object_name.eig_vectors
        """
        if len(self.H.A) == 0:
            print('You have to generate H first')
        if full:
            self.eig_values, self.eig_vectors = np.linalg.eigh(self.H.A)
        else:
            try:
                self.eig_values, self.eig_vectors = scipy.sparse.linalg.eigsh(self.H, k=number_of_states, which='SA')
            except scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence as e:
                print("Didn't manage to solve with sparse solver -> Now trying with np.linalg.eigh")
                self.eig_values, self.eig_vectors = np.linalg.eigh(self.H.A)
