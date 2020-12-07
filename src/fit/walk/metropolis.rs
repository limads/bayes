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

/// A non-parametric representation of a posterior distribution in terms of the sampling
/// trajectory created by a random walk based algorithm, such as the Metropolis-Hastings.
/// To sample from this Distribution, a uniform random sample is taken and interpreted as
/// an index of the random walk, and the values of all parameters at this index are
/// set into the probabilistic graph to yield a result. To calculate the marginal parameter
/// probability, the whole trajectory is averaged, resulting in a histogram for each variable
/// of interest.
#[derive(Debug)]
pub struct RandomWalk {
    /// To retrieve the samples, we need to modify the parameter vector of model
    /// to make use of the sample_into API. But this does not impact the public API,
    /// which is why we use interior mutability here.
    model : RefCell<Model>,
    
    weights : Option<Vec<usize>>
}

impl Marginal<Histogram> for RandomWalk {

    /// Returns a non-parametric representation of this distribution
    /// marginal parameter value at index ix.
    fn marginal(&self, names : &[&str]) -> Option<Histogram> {
        assert!(names.len() == 1);
        //let traj = self.trajectory()?;
        //traj.histogram(ix)
        unimplemented!()
    }
}

impl Predictive for RandomWalk {

    fn predict(&mut self, fixed : Option<&dyn Sample>) -> Box<dyn Sample> {
        unimplemented!()
    }

}

impl Display for RandomWalk {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Random Walk: ()")
    }

}

impl Distribution for RandomWalk
    where Self : Sized
{

    fn set_parameter(&mut self, _p : DVectorSlice<'_, f64>, _natural : bool) {
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

    fn log_prob(&self, _y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>) -> f64 {
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

}

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

// Using as type parameter Estimator<dyn Distribution> and returning
// the dynamic reference Ok(&self.model.into()) triggers a compiler error
// (panic) at nightly (rustc 1.46.0-nightly (346aec9b0 2020-07-11))
impl Estimator<RandomWalk> for Metropolis {

    fn predict<'a>(&'a self, cond : Option<&'a Sample /*<'a>*/ >) -> Box<dyn Sample> {
        unimplemented!()
    }
    
    fn posterior<'a>(&'a self) -> Option<&'a RandomWalk> {
        self.rw.as_ref()
    }
    
    fn fit<'a>(&'a mut self) -> Result<&'a RandomWalk, &'static str> {
        let (n, burn) = (self.n, self.burn);
        let mut param_len = 0;
        let (post_left, post_right) = self.model.factors_mut();
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
            model : ((&mut self.model) as *mut Model) as *mut c_void,
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

        if sample_ok {
            let samples = out.transpose();
            utils::set_external_trajectory(&mut self.model, &samples);
        } else {
            return Err("Sampling failed");
        }
        
        self.rw = Some(RandomWalk{ model : RefCell::new(self.model.clone()), weights : None});
        Ok(self.rw.as_ref().unwrap())
    }

}

#[test]
fn metropolis() {
    let mut b = Bernoulli::new(6, Some(0.5))
        .variables(&["y"])
        .condition(Beta::new(1,1));
    let mut data = HashMap::new();
    data.insert("y", vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    b.observe(&data);
    let mut metr = Metropolis::new(b, Some(Settings{ n : 100, burn : 50, chains : 1 }));
    let post : Result<&RandomWalk, _> = metr.fit();
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
