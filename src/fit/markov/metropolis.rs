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
use crate::sample::Sample;
use std::fmt::{self, Display};
use crate::prob;
use std::collections::HashMap;
use crate::fit::markov::Trajectory;
use rand;
use crate::approx::{RandomWalk, Histogram};

/// Metropolis-Hastings settings.
#[derive(Debug)]
pub struct Settings {

    /// Burn-in: Number of samples discarded before sampling.
    pub burn : usize,

    /// Total number of samples to collect.
    pub n : usize,

    /// How many parallel chains to run.
    pub chains : usize
}

/* The design of Estimator is being violated here.

In conjugate estimation, the distribution the user creates is the likelihood conditioned on the
prior. fit(.) modifies the likelihood, and it is not a likelihood anymore (the data is consumed);
but a Predictive. A predictive has a posterior field (which is the modified prior), the realizations
of which are the basis for generating new predictions, and can be accessed by the user view view_posterior(.)

By the same logic, we should replace the constant parameters in the graph by a Trajectory structure
at each node. The likelihood now is the
predictive, and we require a realization from the RandomWalk to set the value and make the predictions.
But RandomWalk is different because it is a very complex distribution, which bears many relationships
to each node in the graph. The consistent design would be that the user still has access to its
original (predictive) distribution, but now view_posterior(.) returns this RandomWalk; predict(.) samples from
the random walk and sets all parameters in the graph.

The posterior is accessed as a reference from a field owned by predictor. Metropolis<T> here refers
to the posterior predictive; &RandomWalk returned by view_posterior(.) from this predictive is
what should implement posterior.

// User declares model:
likelihood|prior

(...)
likelihood.fit()
(...)

// Now user has:
predictive|posterior

BUT likelihood/predictive are the SAME structure. Before model fitting, the model.view_data(.) returns Some(data)
and model.predict() returns None; after fitting, model.view_data(.) returns None and model.predict(.) returns Some(pred).

For consistency, we could always wrap inference in an algorithm structure:
let conj = Conjugate::new(norm); and then always have this structure be considered the posterior predictive (which holds
the posterior by reference)

Or we could consider Metropolis<D>/Regression<D> to be the posterior predictive, where D is the user's model.

Or we could consider Metropolis<D>/Regression<D> to only be estimators. We will then use:

let metr = Metropolis::new(lik);
metr.fit().unwrap();
let pred = metr.take_model();

And consider pred to be the predictive implementor (which makes more sense, since they are distributions).
In this case, all distributions would hold a trajectory from which they would sample individually.

To preserve the first design (Conjugate), we could rely on type inference of the returning type:

let post : Result<&RandomWalk, _> = lik.fit(Some(alg));

Where alg is some generic input options to the estimation algorithm, which could be an associated type.
The user can use the builder pattern to decide on algorithm settings.

pub struct Estimator<P> for L {
    type Algorithm = (); // This for conjugate
}

When we have GATs, we can have: type Algorithm = impl Into<MyAlgorithm>, implementing the conversion
for alternative specialized implementations or procedures for a given algorithm.

And None would be passed to fit in cases where no settings for the algorithm are expected. But the question is: The
likelihood node should own a RandomWalk somehow. We could make impl Condition<RandomWalk<P>> for L where
P is the prior/posterior node and L is the likelihood node. The nodes are then not conditioned on P,
but on a random walk over P. The UnivariateFactor<P> would have a new node with a trajectory variant.
Then, we could make the result not being a &P, but a P:

let post : Result<RandomWalk, _> = lik.fit(Some(opts));

Where the RandomWalk would be a lightweight referential structure pointing to all trajectories
over the probabilistic graph. The conjugates would then implement Posterior not for Normal, Gamma
etc but to &Normal, &Gamma, etc to preserve reference semantics (which would be lost on sampling-based
algorithms). This design allows the user to make predictions from its first node transparently, simply
calling lik.predict(.) (but it is not a likelihood anymore; but a posterior predictive). The conditioning
structure between the posterior predictive and the likelihood/prior is preserved.

But now we should implement Estimator<RandomWalk> for Normal, Poisson, Bernoulli and MultiNormal separately,
but allowing those distributions to have an arbitrarily complex conditioning structure.

To implement the posterior conditional graph, we can either have trajetory be a field of the
distributions (which tie the structure too much to one specific algorithm) or have Trajectory<Distribution>
where Likelihood : Conditional<Trajectory<Distribution>>. This structure samples from the trajectory first,
then set the parameter at the inner distribution, then samples from it. The conditional posterior is then
nodes of type Trajectory<D> for all D nodes. For other algorithms, we would have other structures. For
example, to solve importance sampling, we would have Importance<D> as conditional posterior nodes.
*/
/// Metropolis-Hastings sampler. The Metropolis rule builds a directed random
/// walk by comparing the unnormalized posterior log-proabability and the samples of a known
/// proposal distribution, which forces the algorithm to stay most of the time on regions with
/// non-negligible probability density. The Metropolis-Hastings algorithm should work for 
/// most posteriors, as long as they don't have a very high dimensionality.
#[derive(Debug)]
pub struct Metropolis {
    model : Model,
    burn : usize,
    n : usize,
    rw : Option<RandomWalk>
}

impl Metropolis {

    pub fn new<M>(model : M, settings : Option<Settings>) -> Self
    where
        M : Into<Model>
    {
        let model : Model = model.into();
        let burn = settings.as_ref().map(|s| s.burn ).unwrap_or(500);
        let n = settings.map(|s| s.n ).unwrap_or(2000);
        Self{ model, burn, n, rw : None }
    }

}

fn mcmclib_metropolis(metr : &mut Metropolis, sample : &dyn Sample) -> Result<(), &'static str> {
    /*metr.model.as_mut().observe_sample(sample);
    let (n, burn) = (metr.n, metr.burn);
    let mut param_len = 0;
    let (post_left, post_right) = metr.model.as_mut().factors_mut();
    if let Some(left) = post_left {
        param_len += utils::param_vec_length(left, param_len);
    }
    if let Some(right) = post_right {
        param_len += utils::param_vec_length(right, param_len);
    }
    println!("Parameter vector length = {}", param_len);
    if param_len == 0 {
        return Err("Probabilistic graph does not have Posterior nodes");
    };
    let mut init_vec = DVector::zeros(param_len);
    
    /// Receives the samples from mcmclib at a tall matrix
    let mut out = DMatrix::zeros(n, param_len);

    let mut distr_ptr = DistrPtr {
        model : ((&mut metr.model) as *mut Model) as *mut c_void,
        lp_func : model_log_prob
    };
    let sample_ok = unsafe { distr_mcmc(
        &mut init_vec[0] as *mut f64,
        &mut out[0] as *mut f64,
        n,
        param_len,
        burn,
        &mut distr_ptr as *mut DistrPtr
    ) };

    let param_dims = match metr.model {
        Model::MN(ref mut m) => graph_dimensions(m),
        Model::Bern(ref mut b) => graph_dimensions(b),
        _ => unimplemented!()
    };
    let trajs : Vec<Trajectory> = param_dims.iter()
        .enumerate()
        .map(|(ix, dim)| {
            let mut curr_ix = &param_dims[0..ix].iter().sum();
            Trajectory::from(out.slice((0, *curr_ix), (out.ncols(), param_dims[ix])) )
        }).collect();
    metr.rw = Some(RandomWalk{ model : metr.model.clone(), weights : None, preds : HashMap::new(), trajs });
    Ok(())*/
    unimplemented!()
}

/// Returns dimensions for each node in the graph
fn graph_dimensions<O>(lik : &mut impl Likelihood<O>) -> Vec<usize>
where
    O : ?Sized
{
    lik.iter_factors_mut()
        .map(|factor| factor.view_parameter(true).nrows() )
        .collect()
}

// Instantiate a vector of trajectories with the informed sampling dimension. This follows
// the natural factor graph iteration order.
fn build_trajectories<O>(lik : &mut impl Likelihood<O>, size : usize) -> Vec<Trajectory>
where
    O : ?Sized
{
    let param_dims : Vec<usize> = graph_dimensions(lik);
    param_dims.iter()
        .map(|p| Trajectory::new(size, *p) )
        .collect()
}

// Build a vector of proposals. The full proposal distribution can be interpreted
// as a block-diagonal multinormal with dimension \sum_i k_i where k_i is the dimension
// of the ith factor. The proposal is setup with the prior values for each factor.
fn build_proposals<O>(lik : &mut impl Likelihood<O>) -> Vec<MultiNormal>
where
    O : ?Sized
{
    lik.iter_factors_mut()
        .map(|factor| {
            let f_len = factor.param_len();
            let mean = factor.mean().clone();
            let cov = factor.cov().unwrap();
            MultiNormal::new(f_len, mean, cov).unwrap()
        } ).collect()
}

/// Walk a single step of the metropolis-hastings algorithm.
/// Essentially, the metropolis algorithm is one of the solutions to sample
/// from an unknown distribution (for which we have access only to "unnormalized"
/// probabilities, that allow us to calculate the relative chance of sampling at one
/// position versus the other.
/// lik : Model under evaluation.
/// prev_lp : Previous log-probability evaluation of the model under evaluation.
/// proposals : MultiNormal used to sample the random-walk step, assumed to have same dimensions as trajectories.
/// trajs : Random walk chains, one per distribution node.
/// draw_count: Vector that keeps track of number of draws at the current space position.
fn metropolis_step<O>(
    lik : &mut impl Likelihood<O>,
    prev_lp : &mut f64,
    proposals : &mut [MultiNormal],
    trajs : &mut [Trajectory],
    draw_count : &mut Vec<usize>
) -> Result<(), String>
where
    O : ?Sized
{

    for (prop, mut traj) in proposals.iter().zip(trajs.iter_mut()) {
        prop.sample_into(traj.step());
    }

    for (traj, factor) in trajs.iter().zip(lik.iter_factors_mut()) {
        factor.set_natural(&mut traj.state().iter());
    }

    let curr_lp : f64 = lik.log_prob().ok_or(format!("Error evaluating log-probability"))?;

    // This acceptance ratio is the same for the normalized and un-normalized target
    // (This is another thing we could verify, by using a posterior with known p(.), like conjugates).
    // If the new sample is more likely in the target distribution, then accept_ratio > 1.0 and the move
    // will be accepted with certainty, since draw \in [0, 1]. If accept ratio < 1.0, then the move will
    // be accepted with probability given by draw.
    let accept_ratio = curr_lp / *prev_lp;

    let draw : f64 = rand::random();

    // We represent a proposal step by changing its mean.
    if draw <= accept_ratio {
        for (mut prop, traj) in proposals.iter_mut().zip(trajs.iter()) {
            prop.natural_mut().tr_copy_from(&traj.state());
        }
        // We could verify here a necessary condition for ergodicity, p(x_t|x_t-1) = p(x_t-1|x_t),
        // by verifying the log-prob of the last sample given the past mean equals the log-prob of
        // the mean given the last sample.
        draw_count.push(1);
        *prev_lp = curr_lp;
    } else {
        if let Some(mut last) = draw_count.last_mut() {
            *last += 1;
        }
    }
    Ok(())
}

/// Returns a vector of trajectories and a vector of how many times each position
/// in the trajectory was resampled.
/// Convergence performance, when using only symmetrical gaussian distributions, will be affected by:
/// (1) Asymetrical posteriors (variance is much bigger or smaller for a few factor parameters);
/// (2) Heavily-correlated posteriors (walks in one direction will explore a too big or too small
/// range of parameter values)
/// (3) High-dimensionality of posterior parameter space (it will take more samples to represent it,
/// since the severity of (1) and (2) will increase with parameter space dimensionality).
/// After resolving the random walk, we should check that the local autocorrelation of the samples falls towards zero.
fn native_metropolis<O : ?Sized>(
    mut lik : impl Likelihood<O>,
    n : usize, 
    burn : usize
) -> Result<(Vec<Trajectory>, Vec<usize>), String> {

    let mut trajs : Vec<Trajectory> = build_trajectories(&mut lik, n + burn);
    let mut proposals = build_proposals(&mut lik);
    
    // Starts evaluating the prior at the currently-set parameter values. This is the MLE of
    // each factor if they were started with observe (for higher-level likelihoods) OR the prior values.
    let mut prev_lp = lik.log_prob().ok_or(format!("Error evaluating log-probability"))?;

    let mut curr_lp = 0.0;
    let mut draw_count : Vec<usize> = Vec::with_capacity(n);
    draw_count.push(1);
    
    for i in 0..(n + burn) {
        metropolis_step(&mut lik, &mut prev_lp, &mut proposals[..], &mut trajs[..], &mut draw_count)?;
    }
    
    let trim_trajs : Vec<_> = trajs.drain(0..)
        .map(|mut traj| traj.close().trim_begin(burn) )
        .collect();
    let trim_draw_count : Vec<_> = draw_count.drain(burn..).collect();
    
    Ok((trim_trajs, trim_draw_count))
}

// Using as type parameter Estimator<dyn Distribution> and returning
// the dynamic reference Ok(&self.model.into()) triggers a compiler error
// (panic) at nightly (rustc 1.46.0-nightly (346aec9b0 2020-07-11)).
// We could also implement Estimator<RandomWalk> for L where L : Likelihood
// to preserve the semantics of the Estimator trait (the trait Estimator<P> for
// L means we can calculate a Posterior P from the model L). est.fit(alg) could
// also receive an algorithm carrying the estimation settings and method, if
// there are multiple ways to achieve Posterior P from L.
impl Estimator<'_, RandomWalk> for Metropolis {

    type Algorithm = ();

    type Error = &'static str;

    // fn predict<'a>(&'a self, cond : Option<&'a Sample /*<'a>*/ >) -> Box<dyn Sample> {
    //    unimplemented!()
    // }
    
    /*fn take_posterior(mut self) -> Option<RandomWalk> {
        self.rw.take()
    }
    
    fn view_posterior<'a>(&'a self) -> Option<&'a RandomWalk> {
        self.rw.as_ref()
    }*/
    
    fn fit<'a>(&'a mut self, algorithm : Option<Self::Algorithm>) -> Result<RandomWalk, &'static str> {
        #[cfg(feature="mcmclib")]
        {
            mcmclib_metropolis(&mut self.metropolis, &sample)?;
            Ok(self.rw.clone().unwrap())
        }

        let ans = match self.model {
            Model::MN(ref mut m) => native_metropolis(m.clone(), self.n, self.burn),
            Model::Bern(ref mut b) => native_metropolis(b.clone(), self.n, self.burn),
            _ => unimplemented!()
        };
        match ans {
            Ok((trajs, draw_count)) => {
                self.rw = Some(RandomWalk{
                    model : self.model.clone(),
                    weights : Some(draw_count),
                    preds : HashMap::new(),
                    trajs
                });
                Ok(self.rw.clone().unwrap())
            },
            Err(e) => {
                println!("{}", e);
                Err("MCMC fitting error")
            }
        }
    }

}

#[test]
fn metropolis() {
    let mut b = Bernoulli::new(6, Some(0.5));
    b.with_variables(&["y"]);
    let mut b = b.condition(Beta::new(1,1));
    let mut data = HashMap::new();
    let v : Vec<f64> = vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    data.insert(format!("y"), v);
    // b.observe(&data);
    let mut metr = Metropolis::new(b, Some(Settings{ n : 100, burn : 50, chains : 1 }));
    let post : Result<&RandomWalk, _> = metr.fit(&data);
    println!("Fitting result: {:?}", post);
}

/*/// (Work in progress) The Metropolis-Hastings posterior sampler generate random walk draws
/// from a known proposal distribution (like a gaussian approximation generated by the
/// EM algorithm) and checks how the log-probability of this draw overestimate or underestimate
/// the unknown target posterior density. The size of the mismatch between the proposal and
/// the target distribution is used to build a decision rule to either re-sample at the current
/// position or move the position at which draws are made. After many iterations,
/// the accumulated samples generate a non-parametric representation
/// of the marginal posterior distribution, from which summary statistics can be calculated
/// by averaging over sufficiently spaced draws.
pub struct Metropolis<D>
    where
        D : Distribution
{

    _model : D,

    _proposal : MultiNormal

}

impl<D> Metropolis<D>
    where D : Distribution
{

    fn _step(&mut self) -> bool {
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

    fn fit<'a>(&'a mut self, _y : DMatrix<f64>, x : Option<DMatrix<f64>>) -> Result<&'a D, &'static str> {
        /*let mut lp = 0.0;
        let f = |post_node : &mut dyn Posterior| {
            f.trajectory_mut().unwrap().step();
        };
        // setting the approximation at a likelihood node should also set
        // the approximation at all factors. Calling log_prob(.) over the
        // approximation returns the log_prob(.) of the approximation plus
        // of each approximation factor recursively.
        let mut approx_lp = 0.0
        let (left_fact, right_fact) = self.dyn_factors();
        if let Some(left_fact) = left_fact {
            approx_lp += left_fact.log_prob(y);
        }
        if let Some(right_fact) = rigth_fact {
            approx_lp += rigth_fact.log_prob(y);
        }
        let target_lp = self.log_prob(y);
        let metr_ratio = ...
        let mut pos = 0;
        let mut weights = DVector::from_element(max_iter, 0. as f64);
        let w_incr = 1. / max_iter as f64;
        for _ in 0..max_iter {
            if metr_ratio >= 1.0 {
                self.visit_factors(f);
                pos += 1;
            }
            weights[pos] += w_incr;
        }
        let weigths = weights.remove_columns(pos, weights.nrows() - pos);*/
        unimplemented!()
    }

}*/
