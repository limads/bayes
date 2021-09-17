use nalgebra::{DVector, DMatrix};

use rand;

mod normal;

mod multinormal;

mod categorical;

pub use normal::*;

pub use multinormal::*;

pub use categorical::*;

pub trait Prior {

    fn prior(loc : f64, scale : Option<f64>) -> Self;

}

pub trait Posterior {

    fn size(&self) -> Option<usize>;

}

/// Implement Joint<Normal, Association=f64> for Normal AND Joint<Categorical, Association=Contingency> for Categorical.
pub trait Joint {

    type Association;

    fn joint<const U : usize>(&mut self, other : [Self; U], assoc : Self::Association)
    where
        Self : Sized;

}

// ScaledExponential implementors can be linked indefinitely by the Conditional
// trait to follow the right-hand path (scale) in a factor tree. This means elements
// are always independent conditioned on the right-hand distribution realization; considering
// the left-hand (location) term a constant. Implemented by Normal, MultiNormal and
// slices of those distributions.
pub trait ScaledExponential {

    fn scale(&self) -> f64;

    fn inverse_scale(&self) -> f64;

}

// Implemented only for univariate distributions. May also be called Univariate.
// If Distribution does not have a scale parameter, scale(&self) always return 1.0.
// Exponential implementors can be linked indefinitely by the Conditional trait to
// follow the left-hand path (location) in a factor tree. This means elements are always
// independent conditioned on the left-hand distribution realization; considering the right-hand
// (scale) term a constant.
pub trait Exponential {

    fn location(&self) -> f64;

    fn scale(&self) -> Option<f64>;

    /// Returns the RHS of the subtraction inside the exponential.
    fn log_partition(&self) -> f64;

    /// Returns 1/log_partition
    fn partition(&self) -> f64;

    fn natural(canonical : f64) -> f64;

    fn canonical(natural : f64) -> f64;

    fn log_prob(&self, y : f64) -> f64 {
        Self::natural(self.location()) * y - self.log_partition()
    }

    /*fn sample<R>(&self, rng: &mut R) -> f64
    where
        R: rand::Rng + ?Sized;*/

}

// We can impl Exponential<Location=f64> for Normal and impl Exponential<Location=Vec<f64>> for [Normal].
// Likewise, we can impl ScaledExponential<Scale=&[f64]> for MarginalNormal, returning the covariance row.

#[derive(Debug)]
pub enum Factor<D, P> {

    Conjugate { conj_factor : P },

    // coefs hold the product of the cross-covariance MLE, X^Ty and autocovariance (X^T X)^-1 (w=(X^T X)^-1X^Ty)
    // OR, for heteroscedastic observations, the weigted versions of those quantities X^T \Sigma_y^-1 y and (X^T \Sigma_y^-1 X)^-1.
    Fixed { coefs : DVector<f64>, random : DVector<f64>, obs_fixed : DMatrix<f64>, curr_fixed : DVector<f64>, coef_factor : MultiNormal },

    // curr : 0 - Use current distribution values; curr >= 1: Use index i-1 at mix vector.
    Mixture { probs : DVector<f64>, mix : Vec<D>, curr : usize, /*prob_factor : Categorical*/ },

    Stochastic { state : DVector<f64>, state_factor : D }

}

impl<D, P> Factor<D, P> {

    fn conjugate_factor(&self) -> Option<&P> {
        match self {
            Self::Conjugate{ ref conj_factor, .. } => Some(conj_factor),
            _ => None
        }
    }

    fn fixed_coefs(&self) -> Option<&DVector<f64>> {
        match self {
            Self::Fixed { ref coefs, .. } => Some(coefs),
            _ => None
        }
    }

    fn fixed_factor(&self) -> Option<&MultiNormal> {
        match self {
            Self::Fixed { ref coef_factor, .. } => Some(coef_factor),
            _ => None
        }
    }

}

trait Factorable<D, P> {

    fn factor(&self) -> &Option<Box<Factor<D, P>>>;

    fn factor_mut(&mut self) -> &mut Option<Box<Factor<D, P>>>;

}

// num_integer::IterBinomial
// num_integer::multinomial
