from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.1.2'
DESCRIPTION = 'A python package for quantum chemistry/physics to manipulate many-body operators and wave functions.'

setup(
    name="quantnbody",
    version=VERSION,
    author="Saad Yalouz, Martin Rafael Gulin, Sajathan Sekaran",
    author_email="yalouzsaad@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    url='https://github.com/SYalouz/QuantNBody',
    license='',
    install_requires=['scipy', 'numpy', 'numba'],  # 'psi4' But it is not on the PyPi!!!!!!
    keywords=['quantum physics and chemistry', 'quantum many-body systems', 'exact diagonalization'],
    classifiers=[
        "Programming Language :: Python :: 3", 
        "License :: OSI Approved :: Apache Software License "
    ]
)