from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'A package that allows to generate Hamiltonians for Fermi-Hubbard models and ab-initio Hamiltonians.'

setup(
    name="quantnbody",
    version=VERSION,
    author="Saad Yalouz, Martin Rafael Gulin, Sajathan Sekaran",
    author_email="<syalouz@unistra.fr>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['scipy', 'numpy', 'numba'],
    keywords=['python', 'quantum chemistry', 'second quantization', 'Ab-initio Hamiltonian', 'Hubbard model', 
              'Householder transformation', 'block Householder transformation'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
