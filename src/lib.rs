//! This create offers composable abstractions to build
//! probabilistic models and inference algorithms
//! operating on those models.
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

/// Traits and implementations for exponential-family probability distributions
/// with support for sampling, summary statistics, and conditioning.
pub mod distr;

/// Algorithm for approximating posteriors with multivariate normals
/// (Expectation Maximization; work in progress).
pub mod optim;

/// Basis transformations useful to model non-linear processes,
/// moslty via bindings to GSL and MKL (work in progress).
pub mod basis;

/// Supports the derivation of optimized decision rules based on comparison
/// of posterior log-probabilities (work in progress).
pub mod decision;

// Structure for expressing models with probabilistic dependencies built at runtime.
// pub mod graph;

/// Full posterior estimation via simulation (Metropolis-Hastings algorithm)
/// and related non-parametric distribution representation (work in progress).
pub mod sim;

/// Auto-generated bindings to Intel MKL (mostly for basis transformation).
#[cfg(feature = "mkl")]
mod mkl;

/// Auto-generated bindings to GSL (mostly for optimization and sampling).
pub mod gsl;

/// Utilities to load and save data into/from dynamically-allocated matrices.
pub mod io;

