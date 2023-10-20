Note on the structure of the package
=============================================================

.. note::
  The package **QuantNBody** is stuctured in 2 subpackages :code:`quantnbody.fermionic` and :code:`quantnbody.bosonic` which are respectively dedicated to fermionic and bosonic systems.
  Each subpackage contains a list of functions (encapsulated in a local :code:`tools`
  module) one can use for creating/manipulating many-body operators/wavefunctions.
  In practice, to access all the functions, one only needs to import the following modules from quantnbody:


  For fermionic systems
    >>>  import quantnbody.fermionic.tools

  For bosonic systems
    >>>  import quantnbody.bosonic.tools


  In the following, we detail the list of relevant callable functions for each type of many-body system.
