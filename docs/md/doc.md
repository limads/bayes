<!-- Build instructions

This documentation file can be built with:

pandoc -s doc.md -o doc.html --latexmathml

-->

# About

The crate `bayes` offers an API to perform statistical analysis and prediction for Rust programs.

# Technical reference

The trait `Exponential` is the main way to interact with distributions. It offers methods to access each probability distribution parameter. You can sample by calling methods from rand_distr::Distribution, which is also implemented for all distributions.



 



