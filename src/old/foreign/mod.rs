/// Rust-to-C API
pub mod export;

/// C++-to-Rust API for the gcem (Generalized constant expression math) library. Since this
/// is a templated header-only C++ library, we write a few calls with the double precision
/// specialization and export a C APU to them here.
pub mod gcem;

/// C++to-Rut API for the mcmclib library
pub mod mcmc;

/// Auto-generated bindings to GSL (mostly for optimization and sampling).
#[cfg(feature="gsl")]
pub(crate) mod gsl;

/*#[cfg(feature="mkl")]
pub(crate) mod mkl;

pub mod ipp;*/
