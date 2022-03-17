# This script compiles python library from pybind.cpp

PYBIND_LOCATION="./pybind11/include/" 
PYTHON_LOCATION="/home/mrgulin/miniconda3/include/python3.9/" 
PYTHON_VERSION="python3.9" 
# Python needs to have developer environment


clang++ -shared -fPIC -std=c++11 -I$PYBIND_LOCATION -I$PYTHON_LOCATION `$PYTHON_VERSION -m pybind11 --includes` pybind.cpp -o Quant_NBody_accelerate.so `$PYTHON_VERSION-config --ldflags`