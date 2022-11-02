# Welcome to the tutorials !

In this folder we gather a series of tutorials/examples of how to use the QuantNBody package for both fermionic and bosonic systems. 

## Notebooks tutorials

In order to learn the how to domesticate the package, we encourage the new users to follow the jupyter noteboks tutorials in the following order:

- *Tuto_FIRST_STEP.ipynb:*

This tutorial explains the basics of the QuantNBody package. Focusing on fermionic systems, 
we explain how the encoding a many-body basis is realized in practice in the code. We also detail how we encode the single-body hopping operators.
The latter being a central tools for the creation of any particle number conserving operator in the code. 

- *Tuto_PLAYING_WITH_STATES.ipynb:*

This second tutorial illustrates how to easily manipulate multi-body states with the help of different illustrative examples. 

- *Tuto_SPIN_AND_AB_INITIO_HAMILTONIANS.ipynb:*

This third tutorials focuses on the construction of different spin operators (e.g. $\hat{S}_2$) and ab initio electronic structure Hamiltonians.
We will show how easily we can build these operators and use them.

- *Tuto_BOSE_HUBBARD.ipynb:*

For those interesting in bosonic systems, we also describe here equivalent features/functions to build operators, see/manipulate wavefunctions.


## Examples of scripts

We also provide a series of python scripts illustrating different types of many-body calculations implemented fully with the QuantNBody packages. The folder are named
