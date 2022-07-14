use rand_distr;
use std::borrow::Borrow;
use rand_distr::{StandardNormal};
use crate::prob::*;
use std::iter::IntoIterator;
use std::default::Default;
pub use rand_distr::Distribution;
use rand::Rng;
use nalgebra::*;

#[derive(Debug)]
pub struct Normal {

    loc : f64,

    scale : f64,

    n : usize,

}

impl Default for Normal {

    fn default() -> Self {
        Normal { loc : 0.0, scale : 1.0, n : 1 }
    }

}

/*impl Normal {

    fn sample<R>(&self, rng: &mut R) -> f64
    where
        R: rand::Rng + ?Sized,
        // StandardNormal : rand_distr::Distribution<f64>
        // StandardNormal : rand::distributions::Distribution<f64>
    {
        use rand::prelude::*;
        let z : f64 = rng.sample(rand_distr::StandardNormal);
        self.scale.sqrt() * (z + self.loc)
    }

}*/

impl Univariate for Normal { }

impl Exponential for Normal {

    fn location(&self) -> f64 {
        self.loc
    }

    fn link(avg : f64) -> f64 {
        avg
    }

    fn link_inverse(avg : f64) -> f64 {
        avg
    }

    fn scale(&self) -> Option<f64> {
        Some(self.scale)
    }

    // fn partition(&self) -> f64 {
    //    unimplemented!()
    // }

    fn log_partition(&self) -> f64 {
        self.loc.powf(2.) / (2.*self.scale) + self.scale.sqrt().ln()
    }

}

impl rand_distr::Distribution<f64> for Normal {

    fn sample<R>(&self, rng: &mut R) -> f64
    where
        R: rand::Rng + ?Sized
    {
        // If S is an integer seed and a, b c are constants,
        // r = (a S + b) % c is a new random variate (divide a S + b by c).
        // If r1, r2 are uniformly-distributed variates, sqrt(-2 log r1) cos(2 \pi r2) is
        // a standard normal variate (Smith, 1997).
        use rand::prelude::*;
        let z : f64 = rng.sample(rand_distr::StandardNormal);
        z * self.scale.sqrt() + self.loc
    }

}

/*impl Prior for Normal {

    fn prior(loc : f64, scale : Option<f64>) -> Self {
        Self { loc, scale : scale.unwrap_or(1.0), n : 1, factor : None }
    }

}*/

/*impl Posterior for Normal {

    fn size(&self) -> Option<usize> {
        Some(self.n)
    }

}*/

/*impl Likelihood for Normal {

    fn likelihood(sample : &[f64]) -> Joint<Self> {
        // let (loc, scale, n) = crate::calc::running::single_pass_sum_sum_sq(sample.iter());
        // Normal { loc, scale, n }
        Joint::<Normal>::from_slice(sample)
    }

}*/

/*/// Condition<MultiNormal> is implemented for [Normal] but not for Normal,
/// which gives the user some type-safety when separating regression AND mixture
/// from conjugate models. A regression model is Condition<MultiNormal> for [Normal],
/// and a mixture model is Condition<Categorical> for [Normal]. For mixture models,
/// the normals are set at their MLE, and we perform inference over the categorical
/// variable.
impl<'a> Condition<MultiNormal> for [Normal] {

    fn condition(&mut self, f : MultiNormal) -> &mut Self {
        unimplemented!()
    }

}*/

const STANDARD_NORMAL : Normal = Normal {
    loc : 0.0,
    scale : 0.0,
    n : 0
};

/*impl rand::distributions::Distribution<f64> for Normal {

    // Can accept ThreadRng::default()
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand::prelude::*;
        let z : f64 = rng.sample(self.sampler);
        self.scale.sqrt() * (z + self.loc)
    }

}*/

// based on stats::dnorm.ipp
fn normal_log_prob(x : f64, mu : f64, stddev : f64) -> f64 {
    std_normal_log_prob((x - mu) / stddev, stddev)
}

// based on stats::dnorm.ipp
fn std_normal_log_prob(z : f64, stddev : f64) -> f64 {
    -0.5 * (2.0*std::f64::consts::PI).ln() - stddev.ln() - z.powf(2.0) / 2.0
}

// Based on the statslib impl
pub(crate) fn multinormal_log_prob(x : &DVector<f64>, mean : &DVector<f64>, cov : &DMatrix<f64>) -> f64 {

    // This term can be computed at compile time if VectorN is used. Or we might
    // keep results into a static array of f64 and just index it with x.nrows().
    let partition = -0.5 * x.nrows() as f64 * (2.0*std::f64::consts::PI).ln();

    let xc = x.clone() - mean;

    let cov_chol = Cholesky::new(cov.clone()).unwrap();

    // x^T S^-1 x
    let mahalanobis = xc.transpose().dot(&cov_chol.solve(&xc));

    partition - 0.5 * (cov_chol.determinant().ln() + mahalanobis)
}

impl Joint<Normal> {

    pub fn mean(&self) -> &DVector<f64> {
        &self.loc
    }

    pub fn probability(&self, x : &DVector<f64>) -> f64 {
        self.log_probability(x).exp()
    }

    pub fn log_probability(&self, x : &DVector<f64>) -> f64 {
        multinormal_log_prob(x, self.mean(), self.scale.as_ref().unwrap())
    }

}

// When generating joint gaussian samples, add a small multiple of the identity \epsilon I to
// the covariance before inverting it, for numerical reasons.