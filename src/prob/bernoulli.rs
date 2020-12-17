use super::*;
// use std::boxed::Box;
// use super::beta::*;
// use serde::{Serialize, Deserialize};
use rand_distr;
use rand;
use crate::fit::walk::*;
// use std::ops::AddAssign;
use std::default::Default;
use std::fmt::{self, Display};
use anyhow;
use serde_json::{self, Value, map::Map};
use crate::model::Model;
use std::convert::{TryFrom, TryInto};
use crate::model;
// use argmin::prelude::*;
use argmin;
use either::Either;
use crate::fit::Estimator;

pub type BernoulliFactor = UnivariateFactor<Beta>;

/// The Bernoulli is the exponential-family distribution
/// used as the likelihood for binary outcomes. Each realization is parametrized
/// by a proportion parameter θ (0.0 ≥ θ ≥ 1.0), whose natural
/// parameter transformation is the logit ln(θ / (1.0 - θ)):
///
///<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
/// <semantics>
///  <mrow>
///   <mi>p</mi>
///   <mrow>
///    <mrow>
///     <mo fence="true" stretchy="false">(</mo>
///     <mrow>
///      <mrow>
///       <mi>y</mi>
///       <mo stretchy="false">∣</mo>
///       <mi>θ</mi>
///      </mrow>
///     </mrow>
///     <mo fence="true" stretchy="false">)</mo>
///    </mrow>
///    <mo stretchy="false">=</mo>
///    <msup>
///     <mi>θ</mi>
///     <mi>y</mi>
///    </msup>
///   </mrow>
///   <msup>
///    <mrow>
///     <mo fence="true" stretchy="false">(</mo>
///     <mrow>
///      <mrow>
///       <mn>1</mn>
///       <mo stretchy="false">−</mo>
///       <mi>θ</mi>
///      </mrow>
///     </mrow>
///     <mo fence="true" stretchy="false">)</mo>
///    </mrow>
///    <mrow>
///     <mn>1</mn>
///     <mo stretchy="false">−</mo>
///     <mi>y</mi>
///    </mrow>
///   </msup>
///  </mrow>
///  <annotation encoding="StarMath 5.0">p( y divides %theta ) = %theta^y ( 1 - %theta )^{ 1 - y }</annotation>
/// </semantics>
/// </math>
///
/// # Example
///
/// ```
/// use bayes::distr::*;
///
/// let n = 1000;
/// let bern = Bernoulli::new(n, None);
/// let y = bern.sample();
///
/// // Maximum likelihood estimate
/// let mle = Bernoulli::mle((&y).into());
///
/// // Bayesian conjugate estimate
/// let mut bern_cond = bern.condition(Beta::new(1,1));
/// bern_cond.fit(y, None);
/// let post : Beta = bern_cond.take_factor().unwrap();
/// assert!(post.mean()[0] - mle.mean()[0] < 1E-3);
/// ```
#[derive(Debug, Clone)]
pub struct Bernoulli {

    theta : DVector<f64>,

    /// Log-prob argument when struct represent a conditional expectation.
    eta : DVector<f64>,

    factor : BernoulliFactor,

    //eta_traj : Option<Trajectory>,

    sampler : Vec<rand_distr::Bernoulli>,

    /// log-partition vector. Has one entry for each value of the eta vector. The log-partition
    /// is a non-linear function of the natural parameter that is subtracted from the eta*y term
    /// to complete the log-likelihood specification.
    log_part : DVector<f64>,

    /// In case this has a beta factor, when updating the parameter, also update this buffer
    /// with [log(theta) log(1-theta)]; and pass it instead of the eta as the log-prob argument.
    suf_theta : Option<DMatrix<f64>>,

    obs : Option<DVector<f64>>,
    
    name : Option<String>,
    
    n : usize

}

impl Bernoulli {

    /// logit informs if the constructor parameter
    /// and any incoming set_parameter calls should be interpreted as the natural parameter.
    /// Bernoullis parametrized by the logit cannot have conjugate beta factors; but can
    /// be conditioned on generic unconstrained continuous parameters such as the ones
    /// generated by a normal distribution.
    pub fn new(n : usize, theta : Option<f64>) -> Self {
        if let Some(t) = theta {
            assert!(t > 0.0 && t < 1.0);
        }
        let mut bern : Bernoulli = Default::default();
        bern.n = n;
        bern.log_part = DVector::zeros(n);
        let theta = DVector::from_element(n, theta.unwrap_or(0.5));
        bern.set_parameter(theta.rows(0,theta.nrows()), false);
        bern
    }

    /*pub fn factorial(n: usize) -> usize {
        if n < 2 {
            1
        } else {
            n * Self::factorial(n - 1)
        }
    }*/

}

impl Conditional<Beta> for Bernoulli {

    fn condition(mut self, b : Beta) -> Bernoulli {
        self.factor = BernoulliFactor::Conjugate(b);
        self.suf_theta = Some(Beta::sufficient_stat(
            self.theta.slice((0, 0), (self.theta.nrows(), 1))
        ));
        // TODO update sampler vector
        self
    }

    fn view_factor(&self) -> Option<&Beta> {
        match &self.factor {
            BernoulliFactor::Conjugate(d) => Some(d),
            _ => None
        }
    }

    fn take_factor(self) -> Option<Beta> {
        match self.factor {
            BernoulliFactor::Conjugate(b) => Some(b),
            _ => None
        }
    }

    fn factor_mut(&mut self) -> Option<&mut Beta> {
        match &mut self.factor {
            BernoulliFactor::Conjugate(b) => Some(b),
            _ => None
        }
    }

}

impl Conditional<MultiNormal> for Bernoulli {

    fn condition(mut self, m : MultiNormal) -> Bernoulli {
        self.factor = BernoulliFactor::CondExpect(m);
        // TODO Update samplers
        // TODO Update eta, mean, etc.
        self.suf_theta = None;
        self
    }

    fn view_factor(&self) -> Option<&MultiNormal> {
        match &self.factor {
            BernoulliFactor::CondExpect(m) => Some(m),
            _ => None
        }
    }

    fn take_factor(self) -> Option<MultiNormal> {
        match self.factor {
            BernoulliFactor::CondExpect(m) => Some(m),
            _ => None
        }
    }

    fn factor_mut(&mut self) -> Option<&mut MultiNormal> {
        match &mut self.factor {
            BernoulliFactor::CondExpect(m) => Some(m),
            _ => None
        }
    }

}

impl ExponentialFamily<U1> for Bernoulli
    where
        Self : Distribution
{

    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        DVector::from_element(y.nrows(), 1.)
    }

    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        assert!(y.ncols() == 1);
        DMatrix::from_element(1, 1, y.sum() / (y.nrows() as f64) )
    }

    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {
        assert!(t.nrows() == 1 && t.ncols() == 1);
        assert!(self.eta.nrows() == 1);
        assert!(self.log_part.nrows() == 1);
        self.eta[0] * t[0] - self.log_part[0]
    }

    fn update_log_partition<'a>(&'a mut self, eta : DVectorSlice<'_, f64>) {
        if self.log_part.nrows() != eta.nrows() {
            self.log_part = DVector::zeros(eta.nrows());
        }
        self.log_part.iter_mut()
            .zip(eta.iter())
            .for_each(|(l,e)| { *l = (1. + e.exp()).ln(); } );
    }

    fn log_partition<'a>(&'a self) -> &'a DVector<f64> {
        &self.log_part
    }

    /*fn update_grad(&mut self, _eta : DVectorSlice<'_, f64>) {
        unimplemented!()
    }

    fn grad(&self) -> &DVector<f64> {
        unimplemented!()
    }*/

    fn link_inverse<S>(eta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
    where S : Storage<f64, Dynamic, U1>
    {
        eta.map(|e| 1. / (1. + (-1.* e).exp() ) )
    }

    fn link<S>(theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
    where S : Storage<f64, Dynamic, U1>
    {
        theta.map(|p| (p / (1. - p)).ln() )
    }

}

impl Likelihood for Bernoulli {

    fn view_variables(&self) -> Option<Vec<String>> {
        self.name.as_ref().map(|name| vec![name.clone()] )
    }
    
    fn factors_mut<'a>(&'a mut self) -> (Option<&'a mut dyn Posterior>, Option<&'a mut dyn Posterior>) {
        (super::univariate_factor(&mut self.factor), None)
    }
    
    fn with_variables(&mut self, vars : &[&str]) -> &mut Self {
        assert!(vars.len() == 1);
        self.name = Some(vars[0].to_string());
        self
    }
    
    fn observe(&mut self, sample : &dyn Sample) {
        //self.obs = Some(super::observe_univariate(self.name.clone(), self.theta.len(), self.obs.take(), sample));
        let mut obs = self.obs.take().unwrap_or(DVector::zeros(self.theta.len()));
        self.n = 0;
        if let Some(name) = &self.name {
            if let Variable::Binary(col) = sample.variable(&name) {
                for (tgt, src) in obs.iter_mut().zip(col) {
                    if *src {
                        *tgt = 1.0;
                    } else {
                        *tgt = 0.0;
                    }
                    self.n += 1;
                }
            }
        }
        self.obs = Some(obs);
        //self
    }
    
    /*fn mle(y : DMatrixSlice<'_, f64>) -> Result<Self, anyhow::Error> {
        let prop = y.sum() / y.nrows() as f64;
        Ok(Self::new(1, Some(prop)))
    }*/

    /*fn mean_mle(y : DMatrixSlice<'_, f64>) -> f64 {
        assert!(y.ncols() == 1);
        let mle = y.iter().fold(0.0, |ys, y| {
            assert!(*y == 0. || *y == 1.); ys + y
        }) / (y.nrows() as f64);
        mle
    }

    fn var_mle(y : DMatrixSlice<'_, f64>) -> f64 {
        let m = Self::mean_mle(y);
        m * (1. - m)
    }*/

    /*fn visit_factors<F>(&mut self, f : F) where F : Fn(&mut dyn Posterior) {
        match self.factor {
            BernoulliFactor::Conjugate(ref mut b) => {
                f(b);
                b.visit_post_factors(&f as &dyn Fn(&mut dyn Posterior));
            },
            BernoulliFactor::CondExpect(ref mut m) => {
                f(m);
                m.visit_post_factors(&f as &dyn Fn(&mut dyn Posterior));
            },
            _ => { }
        }
    }*/

   /*fn factors_mut(&mut self) -> Factors {
        unimplemented!()
        /*match &mut self.factor {
            BernoulliFactor::Conjugate(b) => {
                let factors = Factors::new_empty();
                let factors = factors.aggregate(b);
                b.aggregate_factors(factors)
                //if let (Some(opt_a), _) = factors.as_slice()[0].dyn_factors_mut() {
                //    unimplemented!()
                //} else {
                //    factors
                //}
            },
            BernoulliFactor::CondExpect(m) => {
                unimplemented!()
            },
            _ => Factors::new_empty()
        }*/
        // f.map(|f| f.aggregate_factors(vec![f].into()) ).unwrap_or(Factors::new_empty())
        //if let Some(f) = factors.as_slice().get(0) {
        //    panic!("Unimplemented")
        //} else {
        //    factors
        //}

        //if let Some(f) = factors.as_slice().get(0) {
        //    f.aggregate_factors(factors)
        //} else {
        //    factors
        //}
        // let factors = f_vec.into();
        // factors
        //if let BernoulliFactor::Conjugate(ref mut b) =
        /*match &mut self.factor {
            BernoulliFactor::Conjugate(b) => {
                b.aggregate_factors(factors)
            },
            BernoulliFactor::CondExpect(m) => {
                m.aggregate_factors(factors)
            },
            _ => Factors::new_empty()
        }*/
    }*/

}

impl Estimator<Beta> for Bernoulli {

    //fn predict<'a>(&'a self, cond : Option<&'a Sample/*<'a>*/>) -> Box<dyn Sample /*<'a>*/> {
    //    unimplemented!()
    //}
    
    fn posterior<'a>(&'a self) -> Option<&'a Beta> {
        unimplemented!()
    }
    
    fn fit<'a>(&'a mut self, sample : &'a dyn Sample) -> Result<&'a Beta, &'static str> {
        self.observe(sample);
        // assert!(y.ncols() == 1);
        // assert!(x.is_none());
        match self.factor {
            BernoulliFactor::Conjugate(ref mut beta) => {
                let y = self.obs.clone().unwrap();
                let n = y.nrows() as f64;
                let ys = y.column(0).sum();
                let (a, b) = (beta.view_parameter(false)[0], beta.view_parameter(false)[1]);
                let new_param = DVector::from_column_slice(&[a + ys, b + n - ys]);
                beta.set_parameter(new_param.rows(0, new_param.nrows()), false);
                Ok(&(*beta))
            },
            _ => Err("Distribution does not have a conjugate factor")
        }
    }

}

impl Distribution for Bernoulli
where Self : Sized
{

    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, natural : bool) {
        let eta = if natural {
            p.clone_owned()
        } else {
            Self::link(&p)
        };
        self.theta = Self::link_inverse(&eta);
        self.update_log_partition(eta.rows(0,eta.nrows()));
        self.eta = eta;
        if let Some(ref mut suf) = self.suf_theta {
            *suf = Beta::sufficient_stat(self.theta.slice((0,0), (self.theta.nrows(),1)));
        }
        self.sampler.clear();
        for t in self.theta.iter() {
            self.sampler.push(rand_distr::Bernoulli::new(*t).unwrap());
        }
    }

    fn view_parameter(&self, natural : bool) -> &DVector<f64> {
        match natural {
            true => &self.eta,
            false => &self.theta
        }
    }

    // fn observations(&self) -> Option<&DMatrix<f64>> {
    //    self.obs.as_ref()
    // }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.theta
    }

    fn mode(&self) -> DVector<f64> {
        self.theta.clone()
    }

    fn var(&self) -> DVector<f64> {
        self.theta.component_mul(&self.theta.map(|p| 1. - p))
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        None
    }

    fn cov_inv(&self) -> Option<DMatrix<f64>> {
        None
    }

    fn log_prob(&self, y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>) -> f64 {
        println!("lp = {}; y = {}", self.log_part, y);
        super::univariate_log_prob(
            y,
            x,
            &self.factor,
            &self.view_parameter(true),
            &self.log_part,
            self.suf_theta.clone()
        )
    }

    fn sample_into(&self, mut dst : DMatrixSliceMut<'_,f64>) {
        use rand_distr::{Distribution};
        for (i, _) in self.theta.iter().enumerate() {
            dst[(i,0)] = (self.sampler[i].sample(&mut rand::thread_rng()) as i32) as f64;
        }
    }

}

impl TryFrom<serde_json::Value> for Bernoulli {

    type Error = String;

    fn try_from(val : Value) -> Result<Self, String> {
        crate::model::parse::parse_univariate::<Bernoulli, Beta>(&val, "prop")
    }

}

impl TryFrom<Model> for Bernoulli {

    type Error = String;

    fn try_from(lik : Model) -> Result<Self, String> {
        match lik {
            Model::Bern(b) => Ok(b),
            _ => Err(format!("Object does not have a top-level bernoulli node"))
        }
    }

}

impl<'a> TryFrom<&'a Model> for &'a Bernoulli {

    type Error = String;

    fn try_from(lik : &'a Model) -> Result<Self, String> {
        match lik {
            Model::Bern(b) => Ok(b),
            _ => Err(format!("Object does not have a top-level bernoulli node"))
        }
    }
}

impl Into<serde_json::Value> for Bernoulli {

    fn into(mut self) -> serde_json::Value {
        let mut child = Map::new();
        if let Some(mut obs) = self.obs.take() {
            let obs_vec : Vec<f64> = obs.data.into();
            let obs_value : Value = obs_vec.into();
            child.insert(String::from("obs"), obs_value);
        }
        if let BernoulliFactor::CondExpect(mn) = self.factor {
            let fv : Value = mn.into();
            child.insert(String::from("prop"), fv);
        } else {
            if let BernoulliFactor::Conjugate(beta) = self.factor {
                let bv : Value = beta.into();
                child.insert(String::from("prop"), bv);
            } else {
                child.insert(String::from("prop"), crate::model::vector_to_value(&self.theta));
            }
        }
        // let mut parent = Map::new();
        // parent.insert(String::from("bernoulli"), Value::Object(child));
        Value::Object(child)
    }

}

/*impl Posterior for Bernoulli {

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Posterior>, Option<&mut dyn Posterior>) {
        match &mut self.factor {
            BernoulliFactor::Conjugate(ref mut b) => (Some(b as &mut dyn Posterior), None),
            BernoulliFactor::CondExpect(ref mut m) => (Some(m as &mut dyn Posterior), None),
            _ => (None, None)
        }
    }

    fn set_approximation(&mut self, m : MultiNormal) {
        unimplemented!()
    }

    fn approximate(&self) -> Option<&MultiNormal> {
        unimplemented!()
    }
}*/

/*impl Trajectory for Bernoulli {

    fn current<'a>(&'a self) -> Option<DVectorSlice<'a, f64>> {
        self.eta_traj.as_ref().and_then(|eta_traj| {
            Some(eta_traj.traj.column(eta_traj.pos))
        })
    }

    fn step_by<'a>(&'a mut self, diff_eta : DVectorSlice<'a, f64>, _update : bool) {
        self.eta_traj.as_mut().unwrap().step_increment(diff_eta);
    }

    fn step_to<'a>(&'a mut self, new_eta : Option<DVectorSlice<'a, f64>>, update : bool) {
        if let Some(ref mut eta_traj) = self.eta_traj {
            eta_traj.step(new_eta)
        } else {
            self.eta_traj = Some(EtaTrajectory::new(new_eta.unwrap()));
        }
        if update {
            self.set_parameter(new_eta.unwrap(), true);
        }
    }

    fn marginal(&self) -> Option<Sample> {
        self.eta_traj.as_ref().and_then(|eta_traj| {
            let cols : Vec<DVector<f64>> = eta_traj.traj.clone()
                .column_iter().take(eta_traj.pos)
                .map(|col| Self::link(&col) ).collect();
            let t_cols = DMatrix::from_columns(&cols[..]);
            Some(Sample::new(t_cols))
        })
    }

}*/

impl Default for Bernoulli {

    fn default() -> Self {
        Bernoulli {
            theta : DVector::from_element(1, 0.5),
            eta : DVector::from_element(1, 0.0),
            factor : BernoulliFactor::Empty,
            //eta_traj : None,
            sampler : Vec::new(),
            log_part : DVector::from_element(1, (2.).ln()),
            suf_theta : None,
            obs : None,
            name : None,
            n : 0
        }
    }

}

impl Display for Bernoulli {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bern({})", self.theta.nrows())
    }

}

/*impl Solver<DVector<f64>> for Bernoulli {

    const NAME : &'static str = "Bernoulli";

    // Defines the computations performed in a single iteration.
    fn next_iter(
        &mut self,
        op: &mut OpWrapper<DVector<f64>>,
        state: &IterState<O>
    ) -> Result<ArgminIterData<O>, argmin::core::Error> {
        /*// First we obtain the current parameter vector from the `state` struct (`x_k`).
        let xk = state.get_param();
        // Then we compute the gradient at `x_k` (`\nabla f(x_k)`)
        let grad = op.gradient(&xk)?;
        // Now subtract `\nabla f(x_k)` scaled by `omega` from `x_k` to compute `x_{k+1}`
        let xkp1 = xk.scaled_sub(&self.omega, &grad);
        // Return new paramter vector which will then be used by the `Executor` to update
        // `state`.
        Ok(ArgminIterData::new().param(xkp1))*/
        unimplemented!()
    }
} */

impl argmin::core::ArgminOp for Bernoulli {

    type Param = DVector<f64>;

    type Output = f64;

    type Hessian = ();

    type Jacobian = DVector<f64>;

    type Float = f64;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(self.log_prob(self.obs.as_ref().unwrap().into(), None))
    }

    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, argmin::core::Error> {
        Ok(self.grad(self.obs.as_ref().unwrap().into(), None))
    }
}

