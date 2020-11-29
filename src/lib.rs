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

#![doc(html_logo_url = "https://github.com/limads/bayes/blob/master/assets/bayes-logo.png")]

/// Traits and implementations for exponential-family probability distributions
/// with support for sampling, summary statistics, and conditioning. TODO rename to
pub mod prob;

//pub mod basis;

/// Supports the derivation of optimized decision rules based on comparison
/// of posterior log-probabilities (work in progress). TODO move under bayes::model,
/// which will concentrate model comparison routines.
pub mod decision;

// Auto-generated bindings to Intel MKL (mostly for basis transformation).
// mod mkl;

/// Auto-generated bindings to GSL (mostly for optimization and sampling).
pub mod gsl;

/// Data structures and generic traits to load and save data into/from dynamically-allocated matrices.
pub mod sample;

// Feature extraction traits, structures and algorithms.
// pub mod feature;

/// Abstraction for probability models defined at runtime, such as models
/// parsed from JSON.
pub mod model;

// Perhaps add mod bayes::inference containing the estimation algorithms:
// bayes::fit::graph::SumProduct;
// bayes::fit::optim::{ExpectMax, IRLS};
// bayes::fit::linear::{OLS, WLS, GLS, IRLS};
// bayes::fit::sim::{Metropolis};
// The top-level modules then would be: distr, inference, decision, feature, model, graph, sample.
// distr would contain all model validation/comparison machinery.
pub mod fit;

// #[cfg(feature="api")]
pub mod api;

/// Stochastic processes (functions of time or space built from distribution compositions).
/// TODO move under bayes::prob.
pub mod stochastic;


