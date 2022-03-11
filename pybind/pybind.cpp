#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <math.h>
#include <iostream>

namespace py = pybind11;

inline int make_integer_out_of_bit_vector(py::array_t<int> ref_state){
	py::buffer_info buf_ref_state = ref_state.request();
	int *ptr_ref_state = (int *) buf_ref_state.ptr;
	
	int number = 0;
	int index = 0;
	int ref_state_length = buf_ref_state.shape[0];
    for  (int digit=0; digit < ref_state_length; digit++){
		number += ptr_ref_state[digit] * pow(2, ref_state_length - index - 1);
		index += 1;
		
	}
	
    return number;
}

inline std::tuple<py::array_t<int>,int> new_state_after_sq_fermi_op(bool type_of_op, int index_mode, py::array_t<int>& fock_state){
	/* now type of op is bool and True is creation and False is annihilation */
	
	py::buffer_info buf_fock_state = fock_state.request();
	int *ptr_fock_state = (int *) buf_fock_state.ptr;
	
	int sum_creation_op=0;
	for (int i=0; i<index_mode; i++){
		sum_creation_op += ptr_fock_state[i];
	}
	int coeff_phase = 1 - 2 * (sum_creation_op % 2);
	if (type_of_op){ // Creation operator
		ptr_fock_state[index_mode] += 1;
	}else{
		ptr_fock_state[index_mode] -= 1;
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



inline std::tuple<int,int> build_final_state_ad_a(py::array_t<int>& ref_state, int p, int q, py::array_t<int>& mapping_kappa){
	
	std::tuple<py::array_t<int>,int> ret1 = new_state_after_sq_fermi_op(false, q, ref_state);
	py::array_t<int> state_one = std::get<0>(ret1);
	std::tuple<py::array_t<int>,int> ret2 = new_state_after_sq_fermi_op(true, p, state_one);
	
	py::buffer_info buf_map = mapping_kappa.request();
	int *ptr_map = (int *) buf_map.ptr;
	
	py::array_t<int> state_two = std::get<0>(ret2);
	int kappa = ptr_map[make_integer_out_of_bit_vector(state_two)];
	int p1 = std::get<1>(ret1);
	int p2 = std::get<1>(ret2);
    return std::make_tuple(kappa, p1 * p2);
	
}

std::tuple<int,int> update_a_dagger_a_p_q(py::array_t<int> & ref_state, int p, int q, py::array_t<int>& mapping_kappa){
	py::buffer_info buf_ref_state = ref_state.request();
	int *ptr_ref_state = (int *) buf_ref_state.ptr;
	bool bool1 = (p != q and (ptr_ref_state[q] == 0 or ptr_ref_state[p] == 1));
	bool bool2 = (ptr_ref_state[q] == 1);
//	std::cout << p << ' ' << q << ' ' << ptr_ref_state[q] << ' ' << ptr_ref_state[p] << ' ' << bool1 << ' ' << bool2 << ' ' << std::endl;
	if ((!bool1) && (bool2)){
		return build_final_state_ad_a(ref_state, p, q, mapping_kappa);
	}
	
	return std::make_tuple(-10, -10);	
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
	
	m.def("make_integer_out_of_bit_vector_fast", &make_integer_out_of_bit_vector, "fast implementation of make_integer_out_of_bit_vector in C++");
	
	m.def("new_state_after_sq_fermi_op_fast", &new_state_after_sq_fermi_op, "fast implementation of new_state_after_sq_fermi_op in C++", 
		  py::return_value_policy::move);  
	m.def("build_mapping_fast", &build_mapping, "fast implementation of build_mapping in C++", 
		  py::return_value_policy::move);
	m.def("build_final_state_ad_a_fast", &build_final_state_ad_a, "fast implementation of build_final_state_ad_a in C++", 
		  py::return_value_policy::move);	
	m.def("update_a_dagger_a_p_q_fast", &update_a_dagger_a_p_q, "fast implementation of the first part of update_a_dagger_a_p_q in C++", 
		  py::return_value_policy::copy);
}

