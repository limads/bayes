use nalgebra::*;
use nalgebra::storage::*;
use std::ops::{Index, Mul, Add, AddAssign, MulAssign, SubAssign};
pub mod sampling;
use simba::scalar::SubsetOf;
use sampling::*;
use std::cmp::PartialOrd;

pub mod conv;

#[cfg(feature="mkl")]
pub(crate) mod fft;

#[cfg(feature="mkl")]
pub use fft::*;

#[cfg(feature="gsl")]
pub(crate) mod dwt;

#[cfg(feature="gsl")]
pub use dwt::*;

#[cfg(feature="gsl")]
mod interp;

#[cfg(feature="gsl")]
pub use interp::{Interpolation, Modality};

/// Owned time series data structure.
#[derive(Debug, Clone)]
pub struct Signal<N> 
where
    N : Scalar
{
    buf : DVector<N>
}

impl<'a, N> Signal<N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd,
    f64 : SubsetOf<N>
{

    pub fn len(&self) -> usize {
        self.buf.nrows()
    }
    
    pub fn new_constant(n : usize, value : N) -> Self {
        Self{ buf : DVector::from_element(n, value) }
    }
    
    pub fn full_epoch(&'a self) -> Epoch<'a, N> {
        Epoch{ slice : self.buf.rows(0, self.buf.nrows()), offset : 0 }
    }
    
    pub fn full_epoch_mut(&'a mut self) -> EpochMut<'a, N> {
        EpochMut{ slice : self.buf.rows_mut(0, self.buf.nrows()), offset : 0 }
    }
    
    pub fn epoch(&'a self, start : usize, len : usize) -> Epoch<'a, N> {
        Epoch{ slice : self.buf.rows(start, len), offset : start }
    }
    
    pub fn epoch_mut(&'a mut self, start : usize, len : usize) -> EpochMut<'a, N> {
        EpochMut{ slice : self.buf.rows_mut(start, len), offset : start }
    }
    
    pub fn iter(&self) -> impl Iterator<Item=&N> {
        self.buf.iter()
    }
    
    pub fn iter_mut(&'a mut self) -> impl Iterator<Item=&'a mut N> {
        self.buf.iter_mut()
    }
    
    pub fn downsample_aliased(&mut self, src : &Epoch<'_, N>) {
        let step = src.slice.len() / self.buf.nrows();
        if step == 1 {
            sampling::slices::convert_slice(
                &src.slice.as_slice(),
                self.buf.as_mut_slice()
            );
        } else {
            let ncols = src.slice.ncols();
            assert!(src.slice.len() / step == self.buf.nrows(), "Dimension mismatch");
            sampling::slices::subsample_convert(
                src.slice.as_slice(), 
                self.buf.as_mut_slice(), 
                ncols, 
                step,
                false
            );
        }
    }
    
    // Iterate over epochs of same size.
    // pub fn epochs(&self, size : usize) -> impl Iterator<Item=Epoch<'_, N>> {
    //    unimplemented!()
    // }
    
    /*pub fn downsample_from(&mut self, other : &Self) {
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
    
    // Move this to method of Pyramid.
    pub fn threshold(&self, thr : &Threshold) -> SparseSignal<'_, N> {
        unimplemented!()
    }*/
}

/// Borrowed subset of a signal.
#[derive(Debug, Clone)]
pub struct Epoch<'a, N>
where
    N : Scalar
{
    offset : usize,
    slice : DVectorSlice<'a, N>
}

#[derive(Debug)]
pub struct EpochMut<'a, N>
where
    N : Scalar
{
    offset : usize,
    slice : DVectorSliceMut<'a, N>
}

impl<'a, N> EpochMut<'a, N> 
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd,
    f64 : SubsetOf<N>
{

    pub fn sum(&self) -> N {
        self.slice.sum()
    }
    
    pub fn mean(&self) -> N {
        self.slice.mean()
    }
    
    pub fn max(&self) -> N {
        self.slice.max()
    }
    
    pub fn min(&self) -> N {
        self.slice.min()
    }
    
    pub fn len(&self) -> usize {
        self.slice.len()
    }
    
    pub fn component_add(&mut self, other : &Epoch<N>) {
        self.slice.add_assign(&other.slice);
    }
    
    pub fn component_sub(&mut self, other : &Epoch<N>) {
        self.slice.sub_assign(&other.slice);
        //unimplemented!()
    }
    
    pub fn component_scale(&mut self, other : &Epoch<N>) {
        self.slice.component_mul_mut(&other.slice);
    }
    
    pub fn offset_by(&mut self, scalar : N) {
        self.slice.add_scalar_mut(scalar);
    }
    
    pub fn scale_by(&mut self, scalar : N) {
        // self.slice.scale_mut(scalar); // Only available for owned versions
        self.slice.iter_mut().for_each(|n| *n *= scalar ); 
    }
    
    pub fn iter_mut(&'a mut self) -> impl Iterator<Item=&'a mut N> {
        self.slice.iter_mut()
    }
    
}

impl<'a, N> Epoch<'a, N> 
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd,
    f64 : SubsetOf<N>
{

    pub fn max(&self) -> N {
        self.slice.max()
    }
    
    pub fn min(&self) -> N {
        self.slice.min()
    }
    
    pub fn sum(&self) -> N {
        self.slice.sum()
    }
    
    pub fn mean(&self) -> N {
        self.slice.mean()
    }
    
    pub fn len(&'a self) -> usize {
        self.slice.len()
    }
    
    pub fn iter(&self) -> impl Iterator<Item=&N> {
        self.slice.iter()
    }
    
}

/*impl<'a, M, N> Downsample<Signal<N>> for Epoch<'a, M> 
where
    M : Scalar + Copy,
    N : Scalar + Copy + From<M>
{
    
    fn downsample_aliased(&self, dst : &mut Signal<N>) {
        let step = self.slice.len() / dst.buf.nrows();
        if step == 1 {
            sampling::slices::convert_slice(
                &self.slice.as_slice()[self.offset..],
                dst.buf.as_mut_slice()
            );
        } else {
            let ncols = dst.buf.ncols();
            assert!(self.slice.len() / step == dst.buf.nrows(), "Dimension mismatch");
            sampling::slices::subsample_convert(
                self.slice.as_slice(), 
                dst.buf.as_mut_slice(), 
                ncols, 
                step,
                false
            );
        }
    } 
}*/

impl<N> Index<usize> for Signal<N> 
where
    N : Scalar
{

    type Output = N;

    fn index(&self, ix: usize) -> &N {
        &self.buf[ix]
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

impl<N> Into<Vec<N>> for Signal<N> 
where
    N : Scalar
{
    fn into(self) -> Vec<N> {
        let n = self.buf.nrows();
        unsafe{ self.buf.data.resize(n) }
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
