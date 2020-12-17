use nalgebra::*;
use iter::*;
use nalgebra::storage::*;
use gsl::*;
use crate::feature::signal::ops::*;
use crate::feature::signal::*;
use crate::feature::image::Image;
use crate::feature::signal::dwt::gsl::*;

pub struct Wavelet2D {
    plan : DWTPlan
}

impl Wavelet2D {

    pub fn new(basis : Basis, sz : usize) -> Result<Self, &'static str> {
        Ok(Self { plan : DWTPlan::new(basis, (sz, sz) )? })
    }
}

impl Forward<Image<f64>> for Wavelet2D {
    
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
}

