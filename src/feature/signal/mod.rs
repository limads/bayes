use nalgebra::*;
use nalgebra::storage::*;

pub mod sampling;

// pub mod ops;

use sampling::*;

pub(crate) mod conv;

#[cfg(feature="mkl")]
pub(crate) mod fft;

#[cfg(feature="mkl")]
pub use fft::*;

#[cfg(feature="gsl")]
pub(crate) mod dwt;

#[cfg(feature="gsl")]
pub use dwt::*;

/// Owned time series data structure.
pub struct Signal<N> 
where
    N : Scalar
{
    buf : DVector<N>
}

impl<N> Signal<N>
where
    N : Scalar
{

    pub fn len(&self) -> usize {
        self.buf.nrows()
    }
    
    pub fn new_constant(n : usize, value : N) -> Self {
        Self{ buf : DVector::from_element(n, value) }
    }
    
    // Iterate over epochs of same size.
    // pub fn epochs(&self, size : usize) -> impl Iterator<Item=Epoch<'_, N>> {
    //    unimplemented!()
    // }
    
    pub fn downsample_from(&mut self, other : &Self) {
        unimplemented!()
    }
    
    pub fn downsample_into(&self, other : &mut Self) {
        unimplemented!()
    }
    
    pub fn upsample_from(&mut self, other : &Self) {
        unimplemented!()
    }
    
    pub fn upsample_into(&self, other : &mut Self) {
        unimplemented!()
    }
    
    pub fn threshold(&self, thr : &Threshold) -> SparseSignal<'_, N> {
        unimplemented!()
    }
}

/// Borrowed subset of a signal.
pub struct Epoch<'a, N>
where
    N : Scalar
{
    src : &'a [N],
    offset : usize,
    sz : usize
}

impl<'a, N> Epoch<'a, N> 
where
    N : Scalar
{

    pub fn len(&'a self) -> usize {
        self.src.len()
    }
    
}

impl<'a, M, N> Downsample<Signal<N>> for Epoch<'a, M> 
where
    M : Scalar + Copy,
    N : Scalar + Copy + From<M>
{
    
    fn downsample(&self, dst : &mut Signal<N>) {
        let step = self.src.len() / dst.buf.nrows();
        if step == 1 {
            sampling::slices::convert_slice(
                &self.src[self.offset..],
                dst.buf.as_mut_slice()
            );
        } else {
            let ncols = dst.buf.ncols();
            assert!(self.src.len() / step == dst.buf.nrows(), "Dimension mismatch");
            sampling::slices::subsample_convert(
                self.src, 
                dst.buf.as_mut_slice(), 
                ncols, 
                step,
                false
            );
        }
    }
    
}

impl<N> From<DVector<N>> for Signal<N> 
where
    N : Scalar
{
    fn from(s : DVector<N>) -> Self {
        Self{ buf : s }
    }
}

impl<N> From<Vec<N>> for Signal<N> 
where
    N : Scalar
{
    fn from(s : Vec<N>) -> Self {
        Self{ buf : DVector::from_vec(s) }
    }
}

impl<N> AsRef<[N]> for Signal<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &[N] {
        self.buf.data.as_slice()
    }
}

impl<N> AsMut<[N]> for Signal<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut [N] {
        self.buf.data.as_mut_slice()
    }
}

impl<N> AsRef<Vec<N>> for Signal<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &Vec<N> {
        self.buf.data.as_vec()
    }
}

impl<N> AsMut<Vec<N>> for Signal<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut Vec<N> {
        unsafe{ self.buf.data.as_vec_mut() }
    }
}

impl<N> AsRef<DVector<N>> for Signal<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &DVector<N> {
        &self.buf
    }
}

impl<N> AsMut<DVector<N>> for Signal<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut DVector<N> {
        &mut self.buf
    }
}

pub struct Threshold {

    /// Minimum distance of neighboring values (in number of samples
    /// or symmetrical pixel area). If None, all values satisfying value
    /// will be accepted.
    pub min_dist : Option<usize>,
    
    /// Threshold value.
    value : f64,
    
    /// All neighboring pixels over the area should be smaller by the
    /// specified ratio. If none, no slope restrictions are imposed.
    slope : Option<f64>
}

/// Result of thresholding a signal. This structure carries a (row, col) index and
/// a scalar value for this index.
pub struct SparseSignal<'a, N> 
where
    N : Scalar
{
    src : &'a Signal<N>
}

/// Subset of a SparseSignal
pub struct SparseEpoch<'a, N> 
where
    N : Scalar
{

    /// Source sparse signal
    source : &'a SparseSignal<'a, N>,
    
    /// Which indices we will use from the source
    ixs : Vec<usize>
}
