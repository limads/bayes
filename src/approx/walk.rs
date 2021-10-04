use nalgebra::*;
use crate::prob::*;
use crate::fit::Estimator;
use crate::model::Model;
use std::ffi::c_void;
use std::cell::RefCell;
use std::convert::{TryFrom, TryInto};
use crate::foreign::export::clang::{DistrPtr, model_log_prob};
use crate::foreign::mcmc::distr_mcmc;
use crate::fit::utils;
// use crate::sample::Sample;
use std::fmt::{self, Display};
use crate::prob;
use std::collections::HashMap;
use crate::fit::markov::Trajectory;
use rand;
use crate::approx::Histogram;

/// A non-parametric representation of a posterior distribution in terms of the sampling
/// trajectory created by a random walk based algorithm, such as the Metropolis-Hastings.
/// To sample from this Distribution, a uniform random sample is taken and interpreted as
/// an index of the random walk, and the values of all parameters at this index are
/// set into the probabilistic graph to yield a result. To calculate the marginal parameter
/// probability, the whole trajectory is averaged, resulting in a histogram for each variable
/// of interest. A random walk is to be interpreted as a non-parametric representation of a
/// distribution, in the same way a Histogram or a Density is. Unlike those representations,
/// a random walk can represent distributions in a great number of dimensions, since its
/// representation does not increase exponentially with the distribution dimensionality,
/// but rather stays constant, depending only on its step rule.
#[derive(Debug, Clone)]
pub struct RandomWalk {
    /// To retrieve the samples, we need to modify the parameter vector of model
    /// to make use of the sample_into API. But this does not impact the public API,
    /// which is why we use interior mutability here.
    pub model : Model,

    pub weights : Option<Vec<usize>>,

    pub preds : HashMap<String, Vec<f64>>,

    pub trajs : Vec<Trajectory>
}

impl RandomWalk {

    /// Evaluates the conditional log-probability of the random walk at its current state,
    /// when all nodes in the graph are held fixed at the current state. All
    /// entries after sep_node (which partitions the graph in its natural iteration order)
    /// are fixed at the t-1 random walk state; While the remaining entries are assumed
    /// to be fixed at the t state. This step can be takes as the basis for a literal implementation of the
    /// Gibbs sampler.
    pub fn gibbs_log_prob(&self, sep_node : usize) -> f64 {
        // Fix parameter entries at past nodes
        // self.model.iter_factors_mut()
        //    .zip(self.trajs)
        //    .skip(sep_node).map(|node| node.set_natural( self.state() );

        // Fix parameter entries at current node
        unimplemented!()
    }

}

/*impl Posterior<Histogram> for RandomWalk {

    /// Returns a non-parametric representation of this distribution
    /// marginal parameter value at index ix.
    fn marginal(&self, ix : usize) -> Option<Histogram> {
        // assert!(names.len() == 1);
        // let traj = self.trajectory()?;
        // traj.histogram(ix)
        unimplemented!()
    }
}*/

/*impl Predictive for RandomWalk {

    fn predict<'a>(&'a mut self, fixed : Option<&dyn Sample>) -> Option<&'a dyn Sample> {
        if let Some(fix) = fixed {
            let lik : &mut dyn Likelihood = self.model.as_mut();
            lik.observe_sample(fix);
        }
        self.preds = prob::predict_from_likelihood(self.model.as_mut(), fixed);
        Some(&self.preds)
    }

    fn view_prediction<'a>(&'a self) -> Option<&'a dyn Sample> {
        if !self.preds.is_empty() {
            Some(&self.preds as &'a dyn Sample)
        } else {
            None
        }
    }

}*/

impl Display for RandomWalk {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Random Walk: ()")
    }

}

/*impl Distribution for RandomWalk
    where Self : Sized
{

    fn sample(&self, dst : &mut [f64]) {

    }

    fn set_parameter(&mut self, _p : DVectorSlice<'_, f64>, _natural : bool) {
        unimplemented!()
    }

    fn set_natural<'a>(&'a mut self, eta : &'a mut dyn Iterator<Item=&'a f64>) {
        unimplemented!()
    }

    fn view_parameter(&self, _natural : bool) -> &DVector<f64> {
        unimplemented!()
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        unimplemented!()
    }

    fn mode(&self) -> DVector<f64> {
        unimplemented!()
    }

    fn var(&self) -> DVector<f64> {
        unimplemented!()
    }

    fn joint_log_prob(&self, /*_y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>*/ ) -> Option<f64> {
        unimplemented!()
    }

    fn sample_into(&self, _dst : DMatrixSliceMut<'_,f64>) {
        unimplemented!()
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        None
    }

    fn cov_inv(&self) -> Option<DMatrix<f64>> {
        None
    }

}*/

/*/// A Monte Carlo integration problem allow us to approximate any integral of a product of two functions:
/// \int p(x) f(x) dx where \int p(x) dx = 1.0 is a distribution which integrates to one and
/// from which we can sample from. The integral is approximated by \sum_{i=1}^n 1/n f(x) where the
/// set of x values is taken by sampling p(x). By taking p to be a posterior distribution (for example,
/// a random walk) we can calculate distribution summaries, for example. We can represent posterior MCMC estimation
/// by creating a type RandomWalkMC = MonteCarlo<RandomWalk> or type MarkovChainMC = MonteCarlo<MarkovChain>, since
/// posterior summaries are just the MonteCarlo methodology applied to the non-parametric random walk ergotic
/// representation of a posterior distribution. The function used in this case is the cumulative (proportion) sum, which gives
/// rise to the cumulative (proportion) histogram. MonteCarlo<Bootstrap<Normal>> or MonteCarlo<Jacknife<Gamma>>
/// allows for the bootstrap/jacknife maximum likelihood estimators if we choose adequate steps.
struct MonteCarlo<P>
where P : Distribution
{
    distr : P
}*/


