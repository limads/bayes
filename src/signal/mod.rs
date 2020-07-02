use nalgebra::*;
use nalgebra::storage::*;

// Wrapper type to the Fast-Fourier transforms provided by MKL
#[cfg(feature = "mkl")]
pub mod fft;

#[cfg(feature = "mkl")]
pub use fft::*;

// Wrapper type to the Wavelet transforms provided by GSL
pub mod dwt;

// mod dwt;

pub use dwt::*;

/// Convolution routines.
pub mod conv;

/// Utilities for interpolating time series and surfaces, offered by GSL. (Work in progress)
pub mod interp;

pub mod sampling;

use sampling::*;

/// Generic trait for frequency or spatio/temporal-frequency domain transformations.
/// The input data to the try_forward/forward methods must be a matrix or vector
/// of scalar type M; The input data to the try_backward_to/backward_to methods
/// must be a matrix or vector of scalar type N. The type C prescribes the dimensionality
/// of the input/output pair (either vector or matrix).
///
/// The implementor is
/// assumed to own the forward transform buffer (i.e. the frequency domain data), which
/// is why the forward methods return a reference with lifetime tied to the implementor.
/// The backward buffer (if needed) is assumed to have been pre-allocated by the user, which should opt
/// to use the try_backward_to directly (which takes a mutable reference and returns it in case of success)
/// or the try_backward, which takes ownership of the buffer and returns it to the user in case of
/// success. Implementors should worry only about implementing the try_forward and try_backward_to
/// calls.
pub trait FrequencyBasis<M, N, C>
    where
        M : Scalar,
        N : Scalar,
        C : Dim
{

    /// Apply the forward transform (original domain to frequency domain).
    fn try_forward<'a, S>(
        &'a mut self,
        src : &Matrix<M,Dynamic,C,S>
    ) -> Option<&'a mut Matrix<N,Dynamic,C,VecStorage<N, Dynamic, C>>>
        where
            S : ContiguousStorage<M, Dynamic, C>;

    /// Apply the backward transform (frequency domain to original domain)
    /// by taking a mutable reference to a pre-allocated buffer.
    fn try_backward_to<'a, 'b, /*S,*/ SB>(
        &'a self,
        //src : &'a Matrix<N,Dynamic,C,S>,
        dst : &'b mut Matrix<M,Dynamic,C,SB>
    ) -> Option<&'b mut Matrix<M,Dynamic,C,SB>>
        where
            //S : ContiguousStorage<N, Dynamic, C>,
            SB : ContiguousStorageMut<M, Dynamic, C>;
            //for<'c> Self : 'a ;

    /// Apply the backward transform (frequency domain to original
    /// domain) by taking ownership of a pre-allocated buffer. The
    /// buffer is kept in memory and returned back to the caller in
    /// case of success; or is de-allocated in case of failure.
    fn try_backward<'a, /*S,*/ SB>(
        &'a self,
        //src : &'a Matrix<N,Dynamic,C,S>,
        mut dst : Matrix<M, Dynamic, C, VecStorage<M, Dynamic, C>>
    ) -> Option<Matrix<M,Dynamic,C,VecStorage<M, Dynamic, C>>>
        where
            //S : ContiguousStorage<N, Dynamic, C>,
            SB : ContiguousStorage<M, Dynamic, C>,
            VecStorage<M, Dynamic, C> : ContiguousStorage<M, Dynamic, C>
    {
        self.try_backward_to( /*&src,*/ &mut dst)?;
        Some(dst)
    }

    fn forward<'a, S>(
        &'a mut self,
        src : &Matrix<M,Dynamic,C,S>
    ) -> &'a mut Matrix<N,Dynamic,C,VecStorage<N, Dynamic, C>>
        where
            S : ContiguousStorage<M, Dynamic, C>
    {
        self.try_forward(src).unwrap()
    }

    fn backward_to<'a, 'b, /*S,*/ SB>(
        &'a self,
        //src : &'a Matrix<N,Dynamic,C,S>,
        dst : &'b mut Matrix<M,Dynamic,C,SB>
    ) -> &'b mut Matrix<M,Dynamic,C,SB>
        where
            //S : ContiguousStorage<N, Dynamic, C>,
            SB : ContiguousStorageMut<M, Dynamic, C>,
            //Self: 'b
    {
        self.try_backward_to( /*src,*/ dst).unwrap()
    }

    /// Applies f to the forward transform of src, then transform
    /// back to the original domain.
    fn apply_forward<S, SD, SF, F>(
        &mut self,
        f : F,
        src : &Matrix<M,Dynamic,C,S>,
        mut dst : Matrix<M, Dynamic, C, SD>
    ) -> Option<Matrix<M,Dynamic,C,SD>>
        where
            S : ContiguousStorage<M, Dynamic, C>,
            SD : ContiguousStorageMut<M, Dynamic, C>,
            F : Fn(&mut Matrix<N,Dynamic,C,VecStorage<N, Dynamic, C>>)->Option<()>,
    {
        self.try_forward(src).and_then(|fwd| f(fwd) )?;
        //let fwd = f(ans)?;
        self.try_backward_to( /*&fwd,*/ &mut dst)?;
        Some(dst)
    }

}

/// Signals are types which holds dynamically-allocated
/// data which is read by re-sampling a source slice.
/// The implementor should know how to re-sample this slice
/// by using its own stride information and the user-supplied
/// step. The implementor might read the step and decide
/// at runtime to do a scalar copy/conversion in case of size
/// n>1; or do a vectorized copy/conversion in case of size n=1.
/// Specialized vectorized (simd) calls can be written for all
/// possible (N,M) pairs.
pub trait Signal<M,N>
    where
        M : Into<N>
{

    fn resample(&mut self, src : &[M], step : usize);

}

impl Signal<u8, f64> for DVector<f32> {

    fn resample(&mut self, src : &[u8], step : usize) {
        if step == 1 {
            convert_f32_slice(
                &src,
                self.data.as_mut_slice()
            );
        } else {
            let ncols = self.ncols();
            assert!(src.len() / step == self.nrows(), "Dimension mismatch");
            subsample_convert_f32(&src, self.data.as_mut_slice(), ncols, step, false);
        }
    }

}

impl Signal<u8, f64> for DVector<f64> {

    fn resample(&mut self, src : &[u8], step : usize) {
        if step == 1 {
            convert_f64_slice(
                &src,
                self.data.as_mut_slice()
            );
        } else {
            let ncols = self.ncols();
            assert!(src.len() / step == self.nrows(), "Dimension mismatch");
            subsample_convert_f64(&src, self.data.as_mut_slice(), ncols, step, false);
        }
    }

}

/// Basis reductions for signals (samples with temporal or spatial autocorrelation).
/// FFTs are provided via bindings to Intel MKL (requires that crate is compiled with
/// feature 'mkl'. DWTs are provided via bindings to GSL. Both algorithms
/// are called through a safe generic trait FrequencyBasis at the module root.
/// Also contain interpolation utilities for signals that are not sampled homogeneously,
/// to satisfy the FFT/DWT equal sample spacing restriction.
pub trait Transform<'a, M, N, C>
    where
        M: Scalar,
        N : Scalar,
        C : Dim
{

    fn forward<S>(&'a mut self, s : &Matrix<M, Dynamic, C, S>) -> &'a Matrix<N, Dynamic, C, VecStorage<N, Dynamic, C>>
        where S : ContiguousStorage<M, Dynamic, C>;

    fn backward(&'a mut self) -> &'a Matrix<M, Dynamic, C, VecStorage<M, Dynamic, C>>;

    fn partial_backward<S>(&'a mut self, n : usize) -> MatrixSlice<'a, M, Dynamic, C, U1, Dynamic>;

    fn coefficients(&'a self) -> &'a Matrix<N, Dynamic, C, VecStorage<N, Dynamic, C>>;

    fn coefficients_mut(&'a mut self) -> &'a mut Matrix<N, Dynamic, C, VecStorage<N, Dynamic, C>>;

    fn domain(&'a self) -> Option<&'a Matrix<M, Dynamic, C, VecStorage<M, Dynamic, C>>>;

    fn domain_mut(&'a mut self) -> Option<&'a mut Matrix<M, Dynamic, C, VecStorage<M, Dynamic, C>>>;

}

