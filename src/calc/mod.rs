use std;

// TODO possibly separate into Binary/Continuous/Count

/// Functions applicable to distribution realizations and parameters,
/// possibly executed by applying vectorized instructions.
pub trait Variate {

    fn logit(&self) -> Self;

    fn sigmoid(&self) -> Self;

    fn center(&self, mean : &Self) -> Self;

    fn unscale(&self, inv_scale : &Self) -> Self;

    fn standardize(&self, mean : &Self, inv_scale : &Self) -> Self;

}

impl Variate for f64 {

    fn logit(&self) -> Self {
        1. / (1. + (-1. * (*self)).exp() )
    }

    fn sigmoid(&self) -> Self {
        (*self / (1. - *self)).ln()
    }

    fn center(&self, mean : &Self) -> Self {
        *self - *mean
    }

    fn unscale(&self, inv_scale : &Self) -> Self {
        *self / *inv_scale
    }

    fn standardize(&self, mean : &Self, inv_scale : &Self) -> Self {
        self.center(mean).unscale(inv_scale)
    }

}


