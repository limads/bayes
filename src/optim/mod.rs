// use nalgebra::*;
// use crate::distr::*;

/// Optimization routines, via bindings to GNU GSL.
pub mod gsl;

/// Expectation-maximization algorithm (Work in progress).
pub mod em;

pub use em::*;


