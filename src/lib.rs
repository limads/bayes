//! This create offers composable abstractions to build
//! probabilistic models and inference algorithms.
//!
//! The trait [Distribution](distr/trait.Distribution.html) offer the basic random sampling and
//! calculation of summary statistic functionality for the typical parametric
//! distributions. Implementors of this trait are located at the `distr` module.
//!
//! The trait [Estimator](distr/trait.Estimator.html) offer the `fit` method, which is implemented
//! by the distributions themselves (conjugate inference) and by generic estimation
//! algorithms. Two algorithms will be provided: [ExpectMax](optim/em/struct.ExpectMax.html)
//! (expectation maximization) which returns a gaussian approximation for each node
//! of a generic distribution graph; and [Metropolis](sim/metropolis/struct.Metropolis.html)
//! (Metropolis-Hastings posterior sampler) which returns
//! a non-parametric marginal histogram for each node.

// #![feature(vec_into_raw_parts)]
// #![feature(extern_types)]
// #![feature(is_sorted)]
// #![feature(min_const_generics)] (Perhaps implement MultiNormal<N : usize>)

#![doc(html_logo_url = "https://raw.githubusercontent.com/limads/bayes/master/assets/bayes-logo.png")]

/// Probability distributions used to build models.
pub mod prob;

//pub mod basis;

// Auto-generated bindings to Intel MKL (mostly for basis transformation).
// mod mkl;

/// Data structures and generic traits to load and save data into/from dynamically-allocated matrices.
pub mod sample;

/// Feature extraction traits, structures and algorithms.
pub mod feature;

// Probability models defined at runtime.
pub mod model;

/// Estimation algorithms
pub mod fit;

// Foreign source code to interface with MKL, GSL and mcmclib.
mod foreign;

/// Mathematical functions useful for probability-related calculations
pub mod calc;

/// Non-parametric distribution representations
pub mod approx;
