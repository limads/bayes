use std;

// TODO possibly separate into Binary (logit/sigmoid conversions) / Continuous (standardization conversions)/
// Count (combinatorial calculations). Also, create calc::special to hold the gamma and beta functions.

/// Functions applicable to distribution realizations and parameters,
/// possibly executed by applying vectorized instructions.
pub trait Variate {

    // Convertes a probability in [0-1] to an odds on [-inf, inf], which is a non-linear function.
    fn odds(&self) -> Self;

    // natural logarithm of the odds. Unlike the odds, the logit is a linear function.
    fn logit(&self) -> Self;

    // Converts a logit on [-inf, inf] back to a probability on [0,1]
    fn sigmoid(&self) -> Self;

    fn center(&self, mean : &Self) -> Self;

    fn unscale(&self, inv_scale : &Self) -> Self;

    fn standardize(&self, mean : &Self, inv_scale : &Self) -> Self;

    fn identity(&self) -> Self;

}

impl Variate for f64 {

    fn identity(&self) -> Self {
        *self
    }

    fn odds(&self) -> f64 {
        *self / (1. - *self)
    }

    // The logit or log-odds of a probability p \in [0,1]
    // is the natural log of the ratio p/(1-p)
    fn logit(&self) -> Self {
        let p = if *self == 0.0 {
            f64::EPSILON
        } else {
            *self
        };
        (p / (1. - p)).ln()
    }

    // The sigmoid is the inverse of the logit (or log-odds function).
    // and is given by 1 / (1 + exp(-logit))
    fn sigmoid(&self) -> Self {
        1. / (1. + (-1. * (*self)).exp() )
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

/// Single-iteration univariate statistical quantities.
pub mod running {

    pub fn mean_variance(d : &[f64], unbiased : bool) -> (f64, f64) {
        let n = d.len() as f64;
        let (sum, sum_sq) = d.iter()
            .map(|s| (*s, s.powf(2.)) )
            .fold((0.0, 0.0), |acc, s| (acc.0 + s.0, acc.1 + s.1) );
        let mean = sum / n;
        let var = (sum_sq - sum.powf(2.) / n) / if unbiased { n - 1. } else { n };
        (mean, var)
    }

    /*pub fn mean_absdev(d : &[f64], unbiased : bool) -> (f64, f64) {
        let n = d.len() as f64;
        let (sum, sum_sq) = d.iter()
            .map(|s| (*s, s.powf(2.)) )
            .fold((0.0, 0.0), |acc, s| (acc.0 + s.0, acc.1 + s.1) );
        let mean = sum / n;
        let var = (sum_sq - sum.powf(2.) / n) / if unbiased { n - 1. } else { n };
        (mean, var)
    }*/

}
