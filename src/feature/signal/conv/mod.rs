/// General-purpose iterators over dynamic matrices.
pub mod iter;

// Native discrete convolution.
//pub mod native;

/// Wrapper type to perform discrete convolution by binding against Intel MKL.
#[cfg(feature = "mkl")]
pub mod mkl;

