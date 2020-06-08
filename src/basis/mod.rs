/// Frequency and time/spatial-frequency basis transformation. FFTs are
/// provided via bindings to Intel MKL (requires that crate is compiled with
/// feature 'mkl'. DWTs are provided via bindings to GSL. Both algorithms
/// are called through a safe generic trait FrequencyBasis at the module root.
pub mod frequency;

/// Principal components analysis basis reduction. The PCA is useful
/// for reducing basis with too many dimensions relative to the
/// number of data points. (Work in progress)
pub mod pca;

/// Spline basis expansion algorithm. Splines are local polynomials
/// with smoothness constraints that are useful to approximate
/// non-linear conditional expectations. (Work in progress)
pub mod spline;

/// Polynomial basis expansion to arbitrary degree. A polynomial
/// basis expansion can be seen as a Taylor series approximation
/// to a conditional expectation. (Work in progress)
pub mod polynomial;

/// Utilities for interpolating time series and surfaces, offered by GSL. (Work in progress)
pub mod interp;


