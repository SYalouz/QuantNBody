#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <math.h>
#include <iostream>
#include <vector>

typedef int8_t small_int;

namespace py = pybind11;

std::vector<int> build_mapping(std::vector<std::vector<small_int>> nbody_basis);


class Hold_vectors {
    public:
        std::vector<std::vector<small_int>> nbody_basis;
        std::vector<int> mapping_kappa;
        Hold_vectors (std::vector<std::vector<small_int>> nbody_basis_inp){
            nbody_basis = nbody_basis_inp;
            mapping_kappa = build_mapping(nbody_basis_inp);
        }
        ~Hold_vectors (){
            std::cout << "Destructed object" << std::endl;
        }
};


int binomial(int n, int k) {
    // n choose k
    double res = 1;
    for (int i = 1; i <= k; ++i)
        res = res * (n - k + i) / i;
    return (int)(res + 0.01);
}

inline int make_integer_out_of_bit_vector(std::vector<small_int> ref_state){
	int number = 0;
	int index = 0;
	int ref_state_length = ref_state.size();
    for  (int digit=0; digit < ref_state_length; digit++){
		number += ref_state[digit] * pow(2, ref_state_length - index - 1);
		index += 1;
		
	}
	
    return number;
}

inline std::tuple<std::vector<small_int>,int> new_state_after_sq_fermi_op(bool type_of_op, int index_mode, std::vector<small_int>& fock_state){
	/* now type of op is bool and True is creation and False is annihilation */
	int sum_creation_op=0;
	for (int i=0; i<index_mode; i++){
		sum_creation_op += fock_state[i];
	}
	int coeff_phase = 1 - 2 * (sum_creation_op % 2);
	if (type_of_op){ // Creation operator
		fock_state[index_mode] += 1;
	}else{
		fock_state[index_mode] -= 1;
	}
	return std::make_tuple(fock_state, coeff_phase);
}


std::vector<int> build_mapping(std::vector<std::vector<small_int>> nbody_basis){
	int dim_H = nbody_basis.size();
	int num_digits =nbody_basis[1].size();

	int size_map = pow(2, num_digits);
	std::vector<int> mapping_kappa (size_map, 0);
	
	int number = 0;
	for (int kappa = 0; kappa < dim_H; kappa++){
		number = 0;

		for (int digit = 0; digit < num_digits; digit++){
			number += nbody_basis[kappa][digit] * pow(2, num_digits - digit - 1);
		}
		mapping_kappa[number] = kappa;
		
	}
	return mapping_kappa;
}



inline std::tuple<int,int> build_final_state_ad_a(std::vector<small_int>& ref_state, int p, int q, Hold_vectors& hold_vector){
	
	std::tuple<std::vector<small_int>,int> ret1 = new_state_after_sq_fermi_op(false, q, ref_state);
	std::vector<small_int> state_one = std::get<0>(ret1);
	std::tuple<std::vector<small_int>,int> ret2 = new_state_after_sq_fermi_op(true, p, state_one);
	std::vector<small_int> state_two = std::get<0>(ret2);

	int kappa = hold_vector.mapping_kappa[make_integer_out_of_bit_vector(state_two)];
	int p1 = std::get<1>(ret1);
	int p2 = std::get<1>(ret2);
    return std::make_tuple(kappa, p1 * p2);
	
}

std::tuple<int,int> update_a_dagger_a_p_q(std::vector<small_int> & ref_state, int p, int q, Hold_vectors& hold_vector){
	bool bool1 = (p != q and (ref_state[q] == 0 or ref_state[p] == 1));
	bool bool2 = (ref_state[q] == 1);
	if ((!bool1) && (bool2)){
		return build_final_state_ad_a(ref_state, p, q, hold_vector);
	}
	return std::make_tuple(-10, -10);
}


template <typename T>
py::array_t<T> mkarray_via_buffer(size_t n) {
    return py::array(py::buffer_info(
        nullptr, sizeof(T), py::format_descriptor<T>::format(), 1, {n}, {sizeof(T)}));
}


std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> calculate_sparse_elements(int p, int q, Hold_vectors&hold_vector){
//    std:: cout << &hold_vector << std::endl;
    int sparse_num;
    int dim_H = hold_vector.nbody_basis.size();
    int n_mo = hold_vector.nbody_basis[0].size() / 2;
    int n_electron = 0;
    for (int i=0; i<n_mo * 2; i++){
        n_electron += hold_vector.nbody_basis[0][i]; // We count number of electrons in the first vector
    }
    if (p == q){
        sparse_num = dim_H * n_electron / (n_mo * 2);
    }else{
        sparse_num = binomial(2 * n_mo - 2, n_electron - 1);
    }
    int i = 0;
        std::vector<int> x_list(sparse_num, 0);
        std::vector<int> y_list(sparse_num, 0);
        std::vector<int> value_list(sparse_num, 0);
    if (p == q){
        for (int kappa=0; kappa<dim_H; kappa++){
            if (hold_vector.nbody_basis[kappa][q] == 0){
                continue;
            }
            x_list[i] = kappa;
            y_list[i] = kappa;
            value_list[i] = 1;
            i++;
        }
    }else if ((p/2 == q/2) || ((p - q) % 2 == 0)){
        for (int kappa=0; kappa<dim_H; kappa++){
            std::vector<small_int> ref_state;
            ref_state = hold_vector.nbody_basis[kappa];

            if ((ref_state[q] == 0) || (ref_state[p] == 1)){
                continue;
            }

            auto ret_tuple = build_final_state_ad_a(ref_state, p, q, hold_vector);
            int kappa2= std::get<0>(ret_tuple);
	        int p1p2 = std::get<1>(ret_tuple);
            x_list[i] = kappa;
            y_list[i] = kappa2;

            value_list[i] = p1p2;
            i++;
        }
    }else{
        std::vector<int> empty(0, 0);
        return std::make_tuple(empty, empty, empty);
    }
    return std::make_tuple(x_list, y_list, value_list);
}

/*
void test_function(py::array_t<int> & ref_state, int p, int q, py::array_t<int>& mapping_kappa){
    for(int k=0; k<1132560; k++){
        std::tuple<int,int> test_obj = update_a_dagger_a_p_q(ref_state, p, q, mapping_kappa);
    }
    return std::make_tuple(x_list, y_list, value_list);
}
std::vector<int> test_function(std::vector<int> A, std::vector<int> B, py::array_t<int> D){
    std::vector<int> result (A.size(), 0);
    for (int i = 0; i < A.size(); i++){
        result[i] = A[i] + B[i];
    }

    py::buffer_info buff_D = D.request();
	int *ptr_D = (int *) buff_D.ptr;
    auto E = mkarray_via_buffer<int>(buff_D.shape[0] - 1);
    py::buffer_info buff_E = E.request();
	int *ptr_E = (int *) buff_E.ptr;
    for (int i = 0; i < buff_D.shape[0] - 1; i++){
        ptr_E[i] = ptr_D[i] + 4;
        std::cout << ptr_E[i]<< ", ";

    }
    return result;
}
*/

void test_function(Hold_vectors& obj){
    for (int i=0; i<obj.mapping_kappa.size();i++){
        std::cout << obj.mapping_kappa[i] << ", ";
    }
}

PYBIND11_MODULE(Quant_NBody_accelerate, m){
    m.doc() = "example plugin"; // Optional docstring
	
	m.def("make_integer_out_of_bit_vector_fast", &make_integer_out_of_bit_vector, "fast implementation of make_integer_out_of_bit_vector in C++",
	      py::return_value_policy::move);
	
	m.def("new_state_after_sq_fermi_op_fast", &new_state_after_sq_fermi_op, "fast implementation of new_state_after_sq_fermi_op in C++", 
		  py::return_value_policy::move);
	m.def("build_mapping_fast", &build_mapping, "fast implementation of build_mapping in C++", 
		  py::return_value_policy::move);
	m.def("build_final_state_ad_a_fast", &build_final_state_ad_a, "fast implementation of build_final_state_ad_a in C++", 
		  py::return_value_policy::move);
	m.def("update_a_dagger_a_p_q_fast", &update_a_dagger_a_p_q, "fast implementation of the first part of update_a_dagger_a_p_q in C++", 
		  py::return_value_policy::move);
	m.def("test_function", &test_function, "test in C++",
		  py::return_value_policy::move);
	m.def("calculate_sparse_elements_fast", &calculate_sparse_elements, "implementation of calculate_sparse_elements in C++",
		  py::return_value_policy::move);

    py::class_<Hold_vectors, std::shared_ptr<Hold_vectors>>(m, "CppObject")
        .def(py::init<std::vector<std::vector<small_int>>>(), py::return_value_policy::reference);
}

