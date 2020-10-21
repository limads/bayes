use nalgebra::*;
use crate::distr::*;
use super::*;
use crate::distr::Estimator;
use crate::model::Model;
use super::super::visitors;
use std::ffi::c_void;
use std::cell::RefCell;
use std::convert::{TryFrom, TryInto};
use crate::api::c_api::{DistrPtr, model_log_prob};

#[derive(Debug)]
pub struct Settings {
    pub burn : usize,
    pub n : usize
}

#[derive(Debug)]
pub struct Metropolis {
    model : Model,
    burn : usize,
    n : usize,
    samples : Option<DMatrix<f64>>
}

impl Metropolis {

    pub fn new<M>(model : M, settings : Option<Settings>) -> Self
    where
        M : Into<Model>
    {
        let model : Model = model.into();
        let burn = settings.as_ref().map(|s| s.burn ).unwrap_or(500);
        let n = settings.map(|s| s.n ).unwrap_or(2000);
        Self{ model, burn, n, samples : None }
    }

}

// Using as type parameter Estimator<dyn Distribution> and returning
// the dynamic reference Ok(&self.model.into()) triggers a compiler error
// (panic) at nightly (rustc 1.46.0-nightly (346aec9b0 2020-07-11))
impl<D> Estimator<D> for Metropolis
where
    D : Distribution,
    for<'a> &'a D : TryFrom<&'a Model, Error=String>
{

    fn fit<'a>(&'a mut self, _y : DMatrix<f64>, _x : Option<DMatrix<f64>>) -> Result<&'a D, &'static str> {
        let (n, burn) = (self.n, self.burn);
        let distr : &dyn Distribution = (&self.model).into();
        let lik_len = distr.view_parameter(true).nrows();
        let param_len_cell = RefCell::new(lik_len);
        self.model.visit_factors(|post| {
            *(param_len_cell.borrow_mut()) = visitors::param_vec_length(post);
        });
        let param_len = param_len_cell.into_inner();
        println!("Parameter vector length = {}", param_len);
        assert!(param_len >= 1);
        let mut init_vec = DVector::zeros(param_len);
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
            self.samples = Some(out);
            // Ok(&self.model.into())
            (&self.model).try_into().map_err(|e| { println!("{}", e); "Invalid likelihood" })
        } else {
            Err("Sampling failed")
        }
    }

}

// Linking happens at build.rs file, pointing to [crate-root]/lib/libmcmcwrapper.so
extern "C" {

    fn distr_mcmc(
        init_vals : *const f64,
        out : *mut f64,
        n : usize,
        p : usize,
        burn : usize,
        distr : *mut DistrPtr
    ) -> bool;

}

#[test]
fn metropolis() {
    let mut b = Bernoulli::new(6, Some(0.5));
    let data = DMatrix::from_column_slice(6, 1, &[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    b.set_observations((&data).into());
    let mut metr = Metropolis::new(b, Some(Settings{ n : 100, burn : 50 }));
    let post : Result<&Bernoulli, _> = metr.fit(data, None);
    println!("Fitting result: {:?}", post);
    println!("Resulting metr: {:?}", metr);
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
