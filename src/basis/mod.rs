/// Frequency and time/spatial-frequency basis transformation. FFTs are
/// provided via bindings to Intel MKL (requires that crate is compiled with
/// feature 'mkl'. DWTs are provided via bindings to GSL. Both algorithms
/// are called through a safe generic trait FrequencyBasis at the module root.
pub mod freq;

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
pub mod poly;

/// Utilities for interpolating time series and surfaces, offered by GSL. (Work in progress)
pub mod interp;

/// Load data from generic 8-bit time stream buffers
mod seq;

/// Load data from generic 8-bit image buffers
mod surf;

pub use seq::*;

pub use surf::*;

#[derive(PartialEq)]
pub enum Encoding {
    U8,
    F32,
    F64
}

#[inline(always)]
fn convert_f32_slice(src : &[u8], dst : &mut [f32]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = *s as f32;
    }
}

#[inline(always)]
fn convert_f64_slice(src : &[u8], dst : &mut [f64]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = *s as f64;
    }
}

#[inline(always)]
fn convert_f32_slice_strided(src : &[u8], dst : &mut [f32], cstride : usize) {
    for i in 0..dst.len() {
        dst[i] = src[i*cstride] as f32
    }
}

#[inline(always)]
fn convert_f64_slice_strided(src : &[u8], dst : &mut [f64], cstride : usize) {
    for i in 0..dst.len() {
        dst[i] = src[i*cstride] as f64
    }
}


