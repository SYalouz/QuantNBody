Installation and Running
========================


Required packages
__________________

To run the **QuantNBody** package, you will need first to install the next four packages:

  #. **SciPy** : for installing, see https://scipy.org/install/
  #. **NumPy** : for installing, see https://numpy.org/install/
  #. **Numba** : for installing, see https://numba.readthedocs.io/en/stable/user/installing.html
  #. **Psi4**  : for installing, see https://anaconda.org/psi4/psi4, or https://psicode.org/psi4manual/1.2.1/conda.html


Installation of QuantNBody
______________________________

An easy way to install the QuantNBody package is by using :code:`pip` in your terminal as follows:

    >>> pip install quantnbody

Another alternative is to clone the repository from our `Github page <https://github.com/SYalouz/QuantNBody>`_  and install it in developper mode.
For this, write the following lines in your terminal:

  >>> git clone https://github.com/SYalouz/QuantNBody.git
  >>> cd QuantNBody
  >>> python -m pip install -e .

And dont forget to always :code:`git pull` if you want to get the latest version of the package!
