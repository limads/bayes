use nalgebra::*;
use crate::distr::*;
use super::*;
use crate::optim::*;
use std::ops::AddAssign;
use crate::distr::Estimator;

/// The Metropolis-Hastings posterior sampler implements a transition rule based on
/// how much each new proposal increases the posterior density:
/// If the generated posterior conditional sample increases the density relative
/// to the previous conditional sample, push it to the chain with probability 1;
/// if not, push it to the chain with probability inversely proportional to how much it
/// underestimes the posterior relative to the previous conditional sample, or
/// just push the old sample again otherwise.
/// At each step, generate a sample from a proposal distribution conditional on the
/// last sample (increment the last sample by a zero-centered gaussian). Then, calculate
/// an acceptance ratio, based on the ratio of the unnormalized density f(theta_t)/f(theta_{t-1}).
/// This procedure makes the next step in the random walk be a mixture of a random gaussian and
/// a random gaussian with an offset which, over the long run, reflects the underlying f(.).
/// (1) Draw a sample from u ~ unif[0,1]
/// (2) Calculate the bounded ratio a = min([theta t]/[theta t-1], 1).
/// (3) If u >  a, accumulate the sample theta t. Accumulate sample theta otherwise.
/// A distribution is stationary with respect to a markov chain if the marignal of
/// this distribution with respect to the transition probabilities equals the unconditioned
/// marginal distribution. A sufficient condition for this to hold is that transition
/// probabilities are symmetric (detailed balance; or reversible Markov chain). This condition
/// guarantees that when the conditionals are marginalized over a sample from the Markov chain, then
/// the samples will approach the target distribution in the limit of repeated samples. The metropolis-
/// hastings algorithm add a correction term q(ztau|z*)/q(z*|ztau) to the bounded metropolis ratio
/// so that the transition probabilities will be symmetric (because inverting the distribution also
/// inverts this ratio). This ratio shifts the draw from a multivariate gaussian, where the scales
/// are selected so steps are close to the magnitude as the measured data.
/// So the iteration goes as:
/// (1) Calc log prob of the proposal at the current and past iterations (lq new; lq past)
/// (2) Calc log prob of the unnormalized target at current and past iterations (lp new; lp past)
/// (3) Calc r = min(1, exp( lp(new)/lp(past)*lq(past)/lq(new) ).
/// (4) Draw u~unif[0,1]. Accumulate theta_new to the histogram u < r; Accumulate theta_old otherwise.
pub struct Metropolis<D>
    where
        D : Distribution
{

    model : D,

    proposal : MultiNormal

}

impl<D> Metropolis<D>
    where D : Distribution
{

    fn step(&mut self) -> bool {
        // (1) Let the posterior have a natural vector order
        // (2) Initialize all parameters from a starting distribution
        // (3) For t = 0..T
        //      (3.1) Sample a proposal from a "jumping" distribution as J_t(theta*|theta_{t-1}) (Random walk increment).
        //      (3.2) Calculate the density ratio r = ( p(theta*|y) / J_t(theta*|theta t-1) ) / ( p(theta t-1|y) / J_t(theta t-1 | theta*) )
        //      (3.3) Set theta_t = theta* with probability min(r, 1); Set theta_t = theta_{t-1} otherwise. (Use uniform RNG over 0-1)
        unimplemented!()
    }

}

impl<D> Estimator<D> for Metropolis<D>
    where D : Distribution
{

    fn fit<'a>(&'a mut self, y : DMatrix<f64>) -> Result<&'a D, &'static str> {
        unimplemented!()
    }

}
