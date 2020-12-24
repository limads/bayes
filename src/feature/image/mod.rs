use nalgebra::*;
use nalgebra::storage::*;
use crate::feature::signal::sampling::{self, *};
use std::ops::{Index, Mul, Add, AddAssign, MulAssign, SubAssign};
use simba::scalar::SubsetOf;

#[cfg(feature="mkl")]
mod fft;

#[cfg(feature="mkl")]
pub use fft::*;

#[cfg(feature="gsl")]
pub(crate) mod dwt;

#[cfg(feature="gsl")]
pub use dwt::*;

#[cfg(feature="gsl")]
mod interp;

#[cfg(feature="gsl")]
pub use interp::Interpolation2D;

/// Digital image, represented row-wise.
#[derive(Debug, Clone)]
pub struct Image<N> 
where
    N : Scalar
{
    /// We implement image in terms of a matrix; this wrapping happens because we do not want to use the 
    /// linear operator interpretation of a matrix: we just want to make use of 2D column-oriented indexing.
    buf : DMatrix<N>
}

impl<N> Image<N>
where
    N : Scalar + Copy + RealField
{

    pub fn new_constant(nrows : usize, ncols : usize, value : N) -> Self {
        Self{ buf : DMatrix::from_element(nrows, ncols, value) }
    }
    
    pub fn shape(&self) -> (usize, usize) {
        self.buf.shape()
    }
    
    pub fn full_window<'a>(&'a self) -> Window<'a, N> {
        self.window((0, 0), self.buf.shape())
    }
    
    pub fn window<'a>(&'a self, offset : (usize, usize), sz : (usize, usize)) -> Window<'a, N> {
        Window { win : self.buf.slice(offset, sz), offset }
    }
    
    pub fn window_mut<'a>(&'a mut self, offset : (usize, usize), sz : (usize, usize)) -> WindowMut<'a, N> {
        WindowMut { win : self.buf.slice_mut(offset, sz), offset }
    }
    
    pub fn downsample_aliased<M>(&mut self, src : &Window<M>) 
    where
        M : Scalar + Copy,
        N : Scalar + From<M>
    {
        let (nrows, ncols) = self.buf.shape();
        let step_rows = src.win.nrows() / nrows;
        let step_cols = src.win.ncols() / ncols;
        assert!(step_rows == step_cols);
        sampling::slices::subsample_convert_window(
            src.win.data.as_slice(),
            src.offset,
            src.win.shape(), 
            (nrows, ncols),
            step_rows,
            self.buf.as_mut_slice().chunks_mut(nrows)
        );
    }
    
    pub fn scale_by(&mut self, scalar : N)  {
        self.buf.scale_mut(scalar);
    }
    
    pub fn unscale_by(&mut self, scalar : N)  {
        self.buf.unscale_mut(scalar);
    }
    
    pub fn width(&self) -> usize {
        self.buf.ncols()
    }
    
    pub fn height(&self) -> usize {
        self.buf.nrows()
    }
    
    // pub fn windows(&self) -> impl Iterator<Item=Window<'_, N>> {
    //    unimplemented!()
    // }
    
}

impl<N> AsRef<DMatrix<N>> for Image<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &DMatrix<N> {
        &self.buf
    }
}

impl<N> AsMut<DMatrix<N>> for Image<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut DMatrix<N> {
        &mut self.buf
    }
}

/// Borrowed subset of an image.
#[derive(Debug, Clone)]
pub struct Window<'a, N> 
where
    N : Scalar
{
    offset : (usize, usize),
    win : DMatrixSlice<'a, N>
}

impl<'a, N> Window<'a, N> 
where
    N : Scalar + Mul<Output=N> + MulAssign
{

    /// Creates a window that cover the whole slice src, assuming it represents a square image.
    pub fn from_square_slice(src : &'a [N]) -> Self {
        Self::from_slice(src, (src.len() as f64).sqrt() as usize)
    }
    
    pub fn sub_window(&'a self, offset : (usize, usize), dims : (usize, usize)) -> Window<'a, N> {
        Self{ win : self.win.slice(offset, dims), offset : (self.offset.0 + offset.0, self.offset.1 + offset.1) }
    }
    
    /// Creates a window that cover the whole slice src.
    pub fn from_slice(src : &'a [N], ncols : usize) -> Self {
        Self{ 
            win : DMatrixSlice::from_slice_generic(src, Dynamic::new(src.len() / ncols), 
            Dynamic::new(ncols)), 
            offset : (0, 0) 
        }
    }
    
    pub fn shape(&self) -> (usize, usize) {
        self.win.shape()
    }
    
    pub fn width(&self) -> usize {
        self.win.ncols()
    }
    
    pub fn height(&self) -> usize {
        self.win.nrows()
    }
    
    /*pub fn row_slices(&'a self) -> Vec<&'a [N]> {
        let mut rows = Vec::new();
        for r in (self.offset.0)..(self.offset.0+self.win_sz.0) {
            let begin = self.win_sz.1*r + self.offset.1;
            rows.push(&self.src[begin..begin+self.win_sz.1]);
        }
        rows
    }*/
    
}

#[derive(Debug)]
pub struct WindowMut<'a, N> 
where
    N : Scalar + Copy
{
    offset : (usize, usize),
    win : DMatrixSliceMut<'a, N>
}

impl<'a, N> WindowMut<'a, N> 
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd,
    f64 : SubsetOf<N>
{

    pub fn component_scale(&mut self, other : &Window<N>) {
        self.win.component_mul_mut(&other.win);
    }

}

impl<'a, N> AsRef<DMatrixSlice<'a, N>> for Window<'a, N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &DMatrixSlice<'a, N> {
        &self.win
    }
}

/*impl<'a, M, N> Downsample<Image<N>> for Window<'a, M> 
where
    M : Scalar + Copy,
    N : Scalar + From<M>
{

    fn downsample(&self, dst : &mut Image<N>) {
        // let (nrows, ncols) = dst.shape();
        /*let step_rows = self.win_sz.0 / dst.nrows;
        let step_cols = self.win_sz.1 / dst.ncols;
        assert!(step_rows == step_cols);
        sampling::slices::subsample_convert_window(
            self.src,
            self.offset,
            self.win_sz, 
            (dst.nrows, dst.ncols),
            step_rows,
            dst.buf.as_mut_slice().chunks_mut(dst.nrows)
        );*/
        unimplemented!()
    }
    
}*/

/// Data is assumed to live on the matrix in a column-order fashion, not row-ordered.
impl<N> From<DMatrix<N>> for Image<N> 
where
    N : Scalar
{
    fn from(buf : DMatrix<N>) -> Self {
        /*let (nrows, ncols) = s.shape();
        let data : Vec<N> = s.data.into();
        let buf = DVector::from_vec(data);
        Self{ buf, nrows, ncols }*/
        Self{ buf }
    }
}

/*impl<N> From<(Vec<N>, usize)> for Image<N> 
where
    N : Scalar
{
    fn from(s : (Vec<N>, usize)) -> Self {
        let (nrows, ncols) = (s.1, s.0.len() - s.1);
        Self{ buf : DVector::from_vec(s.0), nrows, ncols  }
    }
}*/

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

/*impl<N> AsRef<Vec<N>> for Image<N> 
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
}*/

/*impl<N> AsRef<DVector<N>> for Image<N> 
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
}*/

/*/// Result of thresholding an image. This structure carries a (row, col) index and
/// a scalar value for this index. Rename to Keypoints.
pub struct SparseImage {

}

/// Subset of a SparseImage
pub struct SparseWindow<'a> {

    /// Source sparse image
    source : &'a SparseImage,
    
    /// Which indices we will use from the source
    ixs : Vec<usize>
}*/

// TODO implement borrow::ToOwned

/*
// Local maxima of the image multiscale transformation
pub struct Keypoint { }

// Object characterized by a set of close keypoints where ordered pairs share close angles. 
pub struct Edge { }

// Low-dimensional approximation of an edge (in terms of lines and curves).
pub struct Shape { }

// Object characterized by a set of shapes.
pub struct Object { }
*/
