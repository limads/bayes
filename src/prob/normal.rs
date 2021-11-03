use rand_distr;
use super::*;
use std::borrow::Borrow;
use rand_distr::{StandardNormal};
use crate::fit::{Likelihood, FixedLikelihood, MarginalLikelihood};
use std::iter::IntoIterator;
use std::default::Default;

pub use rand_distr::Distribution;

#[derive(Debug)]
pub struct Normal {

    // mean
    loc : f64,

    // variance
    scale : f64,

    // Make part of Factor enum.
    // mix : Vec<Box<Normal>>,

    // data, fixed
    // Box<dyn Iterator<Item=f64>> or Box<dyn Iterator<Item=[f64; 5]>>
    // Would require the use of move semantics (e.g. from Vec<.>) and save
    // up on a copy; but we would lose API generality.

    n : usize,

    // Consider storing StaticRc<N, Normal> if the child factor is [Normal; N]. On
    // the impl Condition<Normal> for [Normal; N] we create StaticRc::<Normal, N, N>.
    // Mutability is only allowed if we join all child nodes.
    factor : Option<Box<Factor<Normal, Normal>>>

}

impl Default for Normal {

    fn default() -> Self {
        Normal { loc : 0.0, scale : 1.0, n : 0, factor : None }
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

impl Markov for Normal {

    // fn markov(order : usize) -> Self {
    //    unimplemented!()
    // }

    fn evolve(&mut self, pt : &[f64], transition : impl Fn(&mut Self, &[f64])) {
        transition(self, pt);
    }
}

impl Exponential for Normal {

    fn location(&self) -> f64 {
        self.loc
    }

    fn natural(canonical : f64) -> f64 {
        canonical
    }

    fn canonical(natural : f64) -> f64 {
        natural
    }

    fn scale(&self) -> Option<f64> {
        Some(self.scale)
    }

    fn partition(&self) -> f64 {
        unimplemented!()
    }

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
        self.scale.sqrt() * (z + self.loc)
    }

}

impl Prior for Normal {

    fn prior(loc : f64, scale : Option<f64>) -> Self {
        Self { loc, scale : scale.unwrap_or(1.0), n : 1, factor : None }
    }

}

impl Posterior for Normal {

    fn size(&self) -> Option<usize> {
        Some(self.n)
    }

}

impl Likelihood for Normal {

    type Observation = f64;

    // type Posterior = Self;

    fn likelihood<S, O>(sample : S) -> Self
    where
        S : IntoIterator<Item=O>,
        O : Borrow<Self::Observation>
    {
        let (loc, scale, n) = single_pass_sum_sum_sq(sample);
        Normal { loc, scale, n, factor : None }
    }

}

pub trait Condition<F> {

    fn condition(&mut self, f : F) -> &mut Self;

}

/// Condition<MultiNormal> is implemented for [Normal] but not for Normal,
/// which gives the user some type-safety when separating regression AND mixture
/// from conjugate models. A regression model is Condition<MultiNormal> for [Normal],
/// and a mixture model is Condition<Categorical> for [Normal]. For mixture models,
/// the normals are set at their MLE, and we perform inference over the categorical
/// variable.
impl<'a> Condition<MultiNormal> for [Normal] {

    fn condition(&mut self, f : MultiNormal) -> &mut Self {
        unimplemented!()
    }

}

/// Used to initialize fixed likelihood arrays.
const STANDARD_NORMAL : Normal = Normal {
    loc : 0.0,
    scale : 0.0,
    n : 0,
    factor : None
};

// This requires knowing the sample size at compile time. TODO implement for [Normal],
// which is !Sized.
impl<const N : usize> Likelihood for [Normal; N] {

    type Observation = f64;

    // type Posterior = Self;

    fn likelihood<S, O>(sample : S) -> Self
    where
        S : IntoIterator<Item=O>,
        O : Borrow<Self::Observation>
    {
        let mut norms : [Normal; N] = [STANDARD_NORMAL; N];
        let mut n_sample = 0;
        for (mut n, s) in norms.iter_mut().zip(sample) {
            n.loc = *s.borrow();
            n.scale = f64::INFINITY;
            n_sample += 1;
        }
        if N != n_sample {
            panic!("Likelihood has {} independent samples, but informed sample has {}", N, n_sample);
        }
        norms
    }

}

impl Likelihood for Vec<Normal> {

    type Observation = f64;

    // type Posterior = Self;

    fn likelihood<S, O>(sample : S) -> Self
    where
        S : IntoIterator<Item=O>,
        O : Borrow<Self::Observation>
    {
        let mut norms = Vec::new();
        let mut n_sample = 0;
        for s in sample {
            let mut n : Normal = Default::default();
            n.loc = *s.borrow();
            n.scale = f64::INFINITY;
            norms.push(n);
            n_sample += 1;
        }
        if norms.len() != n_sample {
            panic!("Likelihood has {} independent samples, but informed sample has {}", norms.len(), n_sample);
        }
        norms
    }

}

/*impl Fixed for Normal {

    type Observation = f64;

    type Posterior = MultiNormal;

    fn fixed<S, F, O>(sample : S, fixed : &[impl IntoIterator<Item=F>]) -> Self
    where
        S : IntoIterator<Item=O>,
        O : Borrow<Self::Observation>,
        F : Borrow<f64>
    {
        unimplemented!()
    }

    fn set_fixed(&mut self, fixed : &[impl Borrow<f64>]) {
        unimplemented!()
    }

}*/

/*impl Marginal for Normal {

    type Observation = f64;

    type Posterior = Categorical;

    // Any samples now calculated with respect to this cluster
    fn set_marginal(&mut self, cluster : usize) {
        unimplemented!()
    }

    fn marginal<S, O>(sample : S, clusters : usize) -> Self
    where
        S : IntoIterator<Item=O>,
        O : Borrow<Self::Observation>
    {
        unimplemented!()
    }
}*/

impl Factorable<Normal, Normal> for Normal {

    fn factor(&self) -> &Option<Box<Factor<Normal, Normal>>> {
        &self.factor
    }

    fn factor_mut(&mut self) -> &mut Option<Box<Factor<Normal, Normal>>> {
        &mut self.factor
    }

}

fn single_pass_sum_sum_sq(sample : impl IntoIterator<Item=impl Borrow<f64>>) -> (f64, f64, usize) {
    let mut n = 0;
    let (sum, sum_sq) = sample.into_iter().fold((0.0, 0.0), |accum, d| {
        n += 1;
        let (sum, sum_sq) = (accum.0 + *d.borrow(), accum.1 + d.borrow().powf(2.));
        (sum, sum_sq)
    });
    // let n = data.len() as f64;
    let mean = sum / (n as f64);
    (mean, sum_sq / (n as f64) - mean.powf(2.), n)
}

/*impl rand::distributions::Distribution<f64> for Normal {

    // Can accept ThreadRng::default()
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand::prelude::*;
        let z : f64 = rng.sample(self.sampler);
        self.scale.sqrt() * (z + self.loc)
    }

}*/

