Installation and Running
========================


Required packages
__________________

To run the **QuantNBody** package, you will need first to install the next four packages:

  #. **SciPy** : For installation, see [SciPy's installation page](https://scipy.org/install/)
  #. **NumPy** : For installation, see [NumPy's installation page](https://numpy.org/install/)
  #. **Numba** : For installation, see [Numba's installation page](https://numba.readthedocs.io/en/stable/user/installing.html)
  #. **Psi4**  : For installation, see [Psi4's Anaconda page](https://anaconda.org/psi4/psi4), or [Psi4's official website](https://psicode.org/psi4manual/1.2.1/conda.html)

Installation of QuantNBody
______________________________

An easy way to install the **QuantNBody** package is by using ``pip`` in your terminal as follows:

.. code-block:: bash

   pip install quantnbody

Another alternative is to clone the repository from our `Github page <https://github.com/SYalouz/QuantNBody>`_ and install it in developer mode.
For this, write the following lines in your terminal:

.. code-block:: bash

   git clone https://github.com/SYalouz/QuantNBody.git
   cd QuantNBody
   python -m pip install -e .

And don't forget to always ``git pull`` if you want to get the latest version of the package!
