use nalgebra::*;
use nalgebra::storage::*;
use crate::feature::signal::sampling::{self, *};

#[cfg(feature="mkl")]
mod fft;

#[cfg(feature="mkl")]
pub use fft::*;

#[cfg(feature="gsl")]
pub(crate) mod dwt;

#[cfg(feature="gsl")]
pub use dwt::*;

/// Digital image, represented row-wise.
pub struct Image<N> 
where
    N : Scalar
{
    /// Images are vectors because they are seen as the right operand to linear
    /// operations.
    buf : DVector<N>,
    nrows : usize,
    ncols : usize
}

impl<N> Image<N>
where
    N : Scalar
{

    pub fn new_constant(nrows : usize, ncols : usize, value : N) -> Self {
        Self{ buf : DVector::from_element(nrows*ncols, value), nrows, ncols }
    }
    
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
    
    pub fn full_window<'a>(&'a self) -> Window<'a, N> {
        Window { 
            src : self.buf.as_slice(),
            src_sz : (self.nrows, self.ncols),
            win_sz : (self.nrows, self.ncols),
            offset : (0, 0)
        }
    }
    
    pub fn window<'a>(&'a self, offset : (usize, usize), sz : (usize, usize)) -> Window<'a, N> {
        Window { 
            src : self.buf.as_slice(),
            src_sz : (self.nrows, self.ncols),
            win_sz : sz,
            offset
        }
    }
    
    // pub fn windows(&self) -> impl Iterator<Item=Window<'_, N>> {
    //    unimplemented!()
    // }
    
}

/*impl Downsample for Image {

    type Output;
    
    fn downsample(&self, dst : &mut Self::Output);
    
}*/

/// Borrowed subset of an image.
pub struct Window<'a, N> 
where
    N : Scalar
{
    src : &'a [N],
    src_sz : (usize, usize),
    offset : (usize, usize),
    win_sz : (usize, usize)
}

impl<'a, N> Window<'a, N> 
where
    N : Scalar
{

    pub fn from_slice(src : &'a [N], dims : (usize, usize)) -> Self {
        Self { 
            src,
            src_sz : dims,
            offset : (0, 0),
            win_sz : dims
        }
    }
    
    pub fn shape(&self) -> (usize, usize) {
        self.win_sz
    }
    
    pub fn row_slices(&'a self) -> Vec<&'a [N]> {
        let mut rows = Vec::new();
        for r in (self.offset.0)..(self.offset.0+self.win_sz.0) {
            let begin = self.win_sz.1*r + self.offset.1;
            rows.push(&self.src[begin..begin+self.win_sz.1]);
        }
        rows
    }
    
}

impl<'a, M, N> Downsample<Image<N>> for Window<'a, M> 
where
    M : Scalar + Copy,
    N : Scalar + From<M>
{

    fn downsample(&self, dst : &mut Image<N>) {
        // let (nrows, ncols) = dst.shape();
        let step_rows = self.win_sz.0 / dst.nrows;
        let step_cols = self.win_sz.1 / dst.ncols;
        assert!(step_rows == step_cols);
        sampling::slices::subsample_convert_window(
            self.src,
            self.offset,
            self.win_sz, 
            (dst.nrows, dst.ncols),
            step_rows,
            dst.buf.as_mut_slice().chunks_mut(dst.nrows)
        );
    }
    
}

/// Data is assumed to live on the matrix in a column-order fashion, not row-ordered.
impl<N> From<DMatrix<N>> for Image<N> 
where
    N : Scalar
{
    fn from(s : DMatrix<N>) -> Self {
        let (nrows, ncols) = s.shape();
        let data : Vec<N> = s.data.into();
        let buf = DVector::from_vec(data);
        Self{ buf, nrows, ncols }
    }
}

impl<N> From<(Vec<N>, usize)> for Image<N> 
where
    N : Scalar
{
    fn from(s : (Vec<N>, usize)) -> Self {
        let (nrows, ncols) = (s.1, s.0.len() - s.1);
        Self{ buf : DVector::from_vec(s.0), nrows, ncols  }
    }
}

impl<N> AsRef<[N]> for Image<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &[N] {
        self.buf.data.as_slice()
    }
}

impl<N> AsMut<[N]> for Image<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut [N] {
        self.buf.data.as_mut_slice()
    }
}

impl<N> AsRef<Vec<N>> for Image<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &Vec<N> {
        self.buf.data.as_vec()
    }
}

impl<N> AsMut<Vec<N>> for Image<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut Vec<N> {
        unsafe{ self.buf.data.as_vec_mut() }
    }
}

impl<N> AsRef<DVector<N>> for Image<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &DVector<N> {
        &self.buf
    }
}

impl<N> AsMut<DVector<N>> for Image<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut DVector<N> {
        &mut self.buf
    }
}

/// Result of thresholding an image. This structure carries a (row, col) index and
/// a scalar value for this index.
pub struct SparseImage {

}

/// Subset of a SparseImage
pub struct SparseWindow<'a> {

    /// Source sparse image
    source : &'a SparseImage,
    
    /// Which indices we will use from the source
    ixs : Vec<usize>
}

// TODO implement borrow::ToOwned
