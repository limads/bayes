use nalgebra::*;

/// Basis reductions based on decomposition of the empirical covariance matrix.
/// Those transformations project samples to the orthogonal axis that preserve
/// global variance (PCA); or preserve within/between-class variance (LDA).
pub mod cov;

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

#[derive(PartialEq)]
pub enum Encoding {
    U8,
    F32,
    F64
}

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



