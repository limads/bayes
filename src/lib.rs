/// Generic trait and implementations for exponential-family probability distributions
/// and non-parametric distribution representations (posterior samples and general-purpose histograms).
pub mod distr;

/// Algorithms for approximating the posterior with multivariate normals:
/// Iteratively re-weighted least squares (Unimodal approximation);
/// and Expectation Maximization (Multimodal approximation).
pub mod optim;

/// Basis transformations useful for modelling processes that are non-linear or
/// present heavy dependencies; Moslty via bindings to GSL and MKL.
pub mod basis;

// Module for evaluating the cost of decisions based on pairs of probabilistic models,
// and deriving optimized rules based on comparison of posterior log-probabilities.
pub mod decision;

// Structure for expressing models with probabilistic dependencies built at runtime.
// pub mod graph;

/// Estimator that yield full posterior via simulation (Metropolis-Hastings algorithm).
pub mod sim;

/// Auto-generated bindings to parts of Intel MKL (Math Kernel Library) useful for basis transformation.
#[cfg(feature = "mkl")]
mod mkl;

/// Auto-generated bindings to GSL (Gnu Scientific Library)
pub mod gsl;

