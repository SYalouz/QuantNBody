#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <math.h>
#include <iostream>
#include <vector>

namespace py = pybind11;
class Hold_vectors {
    public:
        std::vector<std::vector<int>> nbody_basis;
        std::vector<int> mapping_kappa;
        Hold_vectors (std::vector<std::vector<int>> nbody_basis_inp, std::vector<int> mapping_kappa_inp){
            nbody_basis = nbody_basis_inp;
            mapping_kappa = mapping_kappa_inp;
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

inline int make_integer_out_of_bit_vector(std::vector<int> ref_state){
	int number = 0;
	int index = 0;
	int ref_state_length = ref_state.size();
    for  (int digit=0; digit < ref_state_length; digit++){
		number += ref_state[digit] * pow(2, ref_state_length - index - 1);
		index += 1;
		
	}
	
    return number;
}

inline std::tuple<std::vector<int>,int> new_state_after_sq_fermi_op(bool type_of_op, int index_mode, std::vector<int>& fock_state){
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
//
//
//	for (int i=0; i<buf_fock_state.shape[0]; i++){
//	    std::cout << ptr_fock_state[i] << ' ';
//	}
//	std::cout << std::endl;
	return std::make_tuple(fock_state, coeff_phase);
}

py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
  py::buffer_info buf1 = input1.request();
  py::buffer_info buf2 = input2.request();

  if (buf1.size != buf2.size) {
    throw std::runtime_error("Input shapes must match");
  }

  /*  allocate the buffer */
  py::array_t<double> result = py::array_t<double>(buf1.size);

  py::buffer_info buf3 = result.request();

  double *ptr1 = (double *) buf1.ptr,
         *ptr2 = (double *) buf2.ptr,
         *ptr3 = (double *) buf3.ptr;
  int X = buf1.shape[0];
  int Y = buf1.shape[1];

  for (size_t idx = 0; idx < X; idx++) {
    for (size_t idy = 0; idy < Y; idy++) {
      ptr3[idx*Y + idy] = ptr1[idx*Y+ idy] + ptr2[idx*Y+ idy];
    }
  }
 
  // reshape array to match input shape
  result.resize({X,Y});

  return result;
}

py::array_t<int> build_mapping(py::array_t<int> nbody_basis){
	py::buffer_info buff_basis = nbody_basis.request();
	int dim_H = buff_basis.shape[0];
	int num_digits = buff_basis.shape[1];
	// std::cout << dim_H << "   " << num_digits << std::endl;
	int size_map = pow(2, num_digits);
	
	
	py::array_t<int> mapping_kappa = py::array_t<int>(size_map);
	py::buffer_info buff_mapping_kappa = mapping_kappa.request();
	
	int *ptr_map = (int *) buff_mapping_kappa.ptr;
	int *ptr_basis = (int *) buff_basis.ptr;
	
	int number = 0;
	int id_end, id_start;
	for (int i = 0; i < size_map; i++){
		ptr_map[i] = 0;
	}
	
	for (int kappa = 0; kappa < dim_H; kappa++){
		number = 0;
		id_start = kappa * num_digits;
		
		
		for (int digit = 0; digit < num_digits; digit++){
			number += ptr_basis[id_start + digit] * pow(2, num_digits - digit - 1);
//			std::cout<<"       "<<num_digits - digit - 1<< "  "<<ptr_basis[digit]<<"  "<<number<<std::endl;
		}
//		std::cout<<kappa<< "  "<<id_start<<"  "<<id_end<<"  "<<number<<std::endl;
		ptr_map[number] = kappa; //kappa
		
	}
	return mapping_kappa;
}



inline std::tuple<int,int> build_final_state_ad_a(std::vector<int>& ref_state, int p, int q, Hold_vectors& hold_vector){
	
	std::tuple<std::vector<int>,int> ret1 = new_state_after_sq_fermi_op(false, q, ref_state);
	std::vector<int> state_one = std::get<0>(ret1);
	std::tuple<std::vector<int>,int> ret2 = new_state_after_sq_fermi_op(true, p, state_one);

	std::vector<int> state_two = std::get<0>(ret2);
	int kappa = hold_vector.mapping_kappa[make_integer_out_of_bit_vector(state_two)];
	int p1 = std::get<1>(ret1);
	int p2 = std::get<1>(ret2);
    return std::make_tuple(kappa, p1 * p2);
	
}

std::tuple<int,int> update_a_dagger_a_p_q(std::vector<int> & ref_state, int p, int q, Hold_vectors& hold_vector){
	bool bool1 = (p != q and (ref_state[q] == 0 or ref_state[p] == 1));
	bool bool2 = (ref_state[q] == 1);
//	std::cout << p << ' ' << q << ' ' << ptr_ref_state[q] << ' ' << ptr_ref_state[p] << ' ' << bool1 << ' ' << bool2 << ' ' << std::endl;
	if ((!bool1) && (bool2)){
		return build_final_state_ad_a(ref_state, p, q, hold_vector);
	}
	return std::make_tuple(-10, -10);
}

// inspired by https://github.com/pybind/pybind11/blob/master/tests/test_numpy_dtypes.cpp --> 151
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
    // py::array_t<int>& mapping_kappa build_mapping(nbody_basis); TODO: What to do with this nbody_basis. Is it better if we copy it to the object  every time?
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
            std::vector<int> ref_state;
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

//void test_function(py::array_t<int> & ref_state, int p, int q, py::array_t<int>& mapping_kappa){
//    for(int k=0; k<1132560; k++){
//        std::tuple<int,int> test_obj = update_a_dagger_a_p_q(ref_state, p, q, mapping_kappa);
//    }
//    return std::make_tuple(x_list, y_list, value_list);
//}
//std::vector<int> test_function(std::vector<int> A, std::vector<int> B, py::array_t<int> D){
//    std::vector<int> result (A.size(), 0);
//    for (int i = 0; i < A.size(); i++){
//        result[i] = A[i] + B[i];
//    }
//
//    py::buffer_info buff_D = D.request();
//	int *ptr_D = (int *) buff_D.ptr;
//    auto E = mkarray_via_buffer<int>(buff_D.shape[0] - 1);
//    py::buffer_info buff_E = E.request();
//	int *ptr_E = (int *) buff_E.ptr;
//    for (int i = 0; i < buff_D.shape[0] - 1; i++){
//        ptr_E[i] = ptr_D[i] + 4;
//        std::cout << ptr_E[i]<< ", ";
//
//    }
//    return result;
//}

void test_function(Hold_vectors& obj){
    for (int i=0; i<obj.mapping_kappa.size();i++){
        std::cout << obj.mapping_kappa[i] << ", ";
    }
}

// py::array_t<double, py::array::f_style> arr({ 3, 5 });

// Creates a macro function that will be called
// whenever the module is imported into python
// 'Quant_NBody_accelerate' is what we 'import' into python.
// 'm' is the interface (creates a py::module object)
//      for which the bindings are created.
//  The magic here is in 'template metaprogramming'
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
        .def(py::init<std::vector<std::vector<int>>, std::vector<int>>(), py::return_value_policy::reference);
}

