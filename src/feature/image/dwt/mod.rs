use nalgebra::*;
use iter::*;
use nalgebra::storage::*;
use gsl::*;
// use crate::feature::signal::ops::*;
use crate::feature::signal::*;
use crate::feature::image::Image;
use crate::feature::signal::dwt::gsl::*;
use crate::feature::image::dwt::dwt::iter::DWTIteratorBase;

/// Two-dimensional wavelet decomposition
pub struct Wavelet2D {
    plan : DWTPlan
}

/// Output of a wavelet decomposition.
#[derive(Clone, Debug)]
pub struct ImagePyramid<N> 
where
    N : Scalar
{
    pyr : DMatrix<N>
}

impl ImagePyramid<f64> {

    pub fn new_constant(n : usize, value : f64) -> Self {
        Self{ pyr : DMatrix::from_element(n, n, value) }
    }
    
    pub fn levels<'a>(&'a self) -> impl Iterator<Item=DMatrixSlice<'a, f64>> {
        DWTIteratorBase::<&'a DMatrix<f64>>::new_ref(&self.pyr)
    }
    
    pub fn levels_mut<'a>(&'a mut self) -> impl Iterator<Item=DMatrixSliceMut<'a, f64>> {
        DWTIteratorBase::<&'a mut DMatrix<f64>>::new_mut(&mut self.pyr)
    }
}

impl<N> AsRef<[N]> for ImagePyramid<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &[N] {
        self.pyr.data.as_slice()
    }
}

impl<N> AsMut<[N]> for ImagePyramid<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut [N] {
        self.pyr.data.as_mut_slice()
    }
}

impl<N> From<DMatrix<N>> for ImagePyramid<N> 
where
    N : Scalar
{
    fn from(s : DMatrix<N>) -> Self {
        Self{ pyr : s }
    }
}

impl<N> AsRef<DMatrix<N>> for ImagePyramid<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &DMatrix<N> {
        &self.pyr
    }
}

/*impl<N> From<Vec<N>> for Pyramid<N> 
where
    N : Scalar
{
    fn from(s : Vec<N>) -> Self {
        Self{ buf : DVector::from_vec(s) }
    }
}*/

impl Wavelet2D {

    pub fn new(basis : Basis, sz : usize) -> Result<Self, &'static str> {
        Ok(Self { plan : DWTPlan::new(basis, (sz, sz) )? })
    }
    
    pub fn forward_mut(&self, src : &Image<f64>, dst : &mut ImagePyramid<f64>) {
        self.plan.apply_forward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    pub fn forward(&self, src : &Image<f64>) -> ImagePyramid<f64> {
        let (nrows, ncols) = self.plan.shape();
        let mut dst = ImagePyramid::new_constant(nrows, 0.0);
        self.forward_mut(src, &mut dst);
        dst
    }
    
    pub fn backward_mut(&self, src : &ImagePyramid<f64>, dst : &mut Image<f64>) {
        self.plan.apply_backward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    pub fn backward(&self, src : &ImagePyramid<f64>) -> Image<f64> {
        let (nrows, ncols) = self.plan.shape();
        let mut dst = Image::new_constant(nrows, ncols, 0.0);
        self.backward_mut(src, &mut dst);
        dst
    }
}

/*impl Forward<Image<f64>> for Wavelet2D {
    
    type Output = Image<f64>;
    
    fn forward_mut(&self, src : &Image<f64>, dst : &mut Self::Output) {
        self.plan.apply_forward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    fn forward(&self, src : &Image<f64>) -> Self::Output {
        let (nrows, ncols) = self.plan.shape();
        let mut dst = Image::new_constant(nrows, ncols, 0.0);
        self.forward_mut(src, &mut dst);
        dst
    }
}

impl Backward<Image<f64>> for Wavelet2D {
    
    type Output = Image<f64>;
    
    fn backward_mut(&self, src : &Image<f64>, dst : &mut Self::Output) {
        self.plan.apply_backward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    fn backward(&self, src : &Image<f64>) -> Self::Output {
        let (nrows, ncols) = self.plan.shape();
        let mut dst = Image::new_constant(nrows, ncols, 0.0);
        self.backward_mut(src, &mut dst);
        dst
    }
}*/

