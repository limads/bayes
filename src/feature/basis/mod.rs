use nalgebra::*;
use nalgebra::storage::*;

/// Basis reductions based on decomposition of the empirical covariance matrix.
/// Those transformations project samples to the orthogonal axis that preserve
/// global variance (PCA); or preserve within/between-class variance (LDA).
pub mod cov;

/// Basis reductions based on frequency-domain expansions such as the Fourier
/// and Wavelet decompositions.
pub mod freq;

/// Basis reductions for signals (samples with temporal or spatial autocorrelation).
/// FFTs are provided via bindings to Intel MKL (requires that crate is compiled with
/// feature 'mkl'. DWTs are provided via bindings to GSL. Both algorithms
/// are called through a safe generic trait FrequencyBasis at the module root.
/// Also contain interpolation utilities for signals that are not sampled homogeneously,
/// to satisfy the FFT/DWT equal sample spacing restriction.
pub trait Basis<'a, M, N, C>
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

/*/// Polynomial basis expansion to arbitrary degree, eiter globally (Polynomial)
/// or locally (Spline). A polynomial basis expansion can be seen as a nth
/// degree Taylor series approximation to a conditional expectation.
/// Spline basis expansion are similar, but defined more locally over a domain,
/// but have smoothness constraints at the region boundaries, and can be used to
/// build flexible non-linear conditional expectations. (Work in progress)
pub mod poly;*/

// Load data from generic 8-bit time stream buffers
// mod seq;

// Load data from generic 8-bit image buffers
// mod surf;

// pub use seq::*;

// pub use surf::*;

/*#[derive(PartialEq)]
pub enum Encoding {
    U8,
    F32,
    F64
}*/

/*/// Generic basis transformation trait.
pub trait Transform<N, C>
    where
        N : Scalar,
        C : Dim,
        Coefficients : Iterator<Item = Matrix<N, Dynamic, C, SliceStorage<N, Dynamic, C>>>
{

    type Coefficients;

    /// Updates the basis coefficients with the informed data.
    fn update(&mut self, dt : Matrix<f64, Dynamic, C, S>);

    /// Iterate over column vectors of V (PCA;LDA) or over complex
    /// sinusoids (FFT) or over wavelets (DWT).
    fn basis(&self) -> Vec<Matrix<N, Dynamic, C, SliceStorage<N, Dynamic, C>>>;

    /// Iterate over eigenvalues of the decomposition (PCA/LDA) or over
    /// complex coefficients at a vector or matrix (FFT) or over the real
    /// coefficient windows (DWT). Returns single result for PCA/LDA (Option).
    /// Return groups of coefficients for splines, depending on the region
    /// the spline is centered at (Vec).
    fn coefficients(&self) -> Coefficients;

    /// Return an iterator over mutable coefficient groups.
    fn coefficients_mut(&mut self) -> Vec<Matrix<N, Dynamic, C, SliceStorageMut<N, Dynamic, C>>>;

}

pub struct FFT {

    back : Vec<DMatrix<f64>>
}

impl FFT {

    /// Self will decompose the signal at windows of size len.
    /// Signal is shift-invariant within windows at the cost
    /// of reduced spatial resolution. Larger window sizes
    /// increase spatial resolution at each window at the cost
    /// of not being able to examine short-scale temporal
    /// changes. After setting the window, take FFT only of the
    /// updated window, leaving past data at their old state.
    pub fn update_window(&mut self, len : usize) {

    }

    pub fn update_all(&self) {

    }
}

/// Trait shared by basis transformations that expands the dimensionality
/// of input samples (poly::Polynomial; poly::Spline, poly::Interpolation)
pub trait Expansion<C>
    where
        Self : Transform<C>,
        C : Dim
{

    /// Recover back the original function at a higher dimensionality
    /// by expanding the original signal given the informed parameters.
    fn expand(&mut self) -> DMatrixSlice<'_, f64>;

}

/// Trait shared by basis transformations that reduce dimensionality
/// of input (eigen::PCA; eigen::LDA)
pub trait Reduction<C>
    where
        Self : Transform<C>,
        C : Dim
{

    /// Recover back the original function at a lower dimensionality by
    /// reducing the signal using the basis from ix up to length.
    fn reduce(&mut self, ix : usize, length : usize) -> DMatrixSlice<'_, f64>;

}*/

/*pub trait Basis<B, C, R> {

    fn coefficients() -> C;

    fn basis() -> B;

    fn recover(from : usize, to : usize) -> R;
}*/

/*/// Trait shared by all feature extraction algorithms.
pub trait Feature<N, C> {

    fn set_extraction<F>(f : F)
        where F : Fn(DMatrix<N>)->DMatrix<N>;

    fn extract(&mut self) -> DMatrix<N>;

}*/


