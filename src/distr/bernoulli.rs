use super::*;
// use std::boxed::Box;
// use super::beta::*;
// use serde::{Serialize, Deserialize};
use rand_distr;
use rand;
use crate::sim::*;
// use std::ops::AddAssign;
use std::default::Default;

pub type BernoulliFactor = UnivariateFactor<Beta>;

/// The Bernoulli is the exponential-family distribution
/// used as the likelihood for binary outcomes. Each realization is parametrized
/// by a proportion parameter θ (0.0 ≥ θ ≥ 1.0), whose natural
/// parameter transformation is the logit ln(θ / (1.0 - θ))
///
/// # Example
///
/// ```
/// use rand;
/// use bayes::distr::*;
/// use nalgebra::*;
///
/// let n = 1000;
/// let mut y = DMatrix::<f64>::zeros(n, 1);
/// y.iter_mut().for_each(|b| *b = (rand::random::<bool>() as i32) as f64);
///
/// // Maximum likelihood estimate
/// assert!(Bernoulli::mean_mle((&y).into()) - 0.5 < 1E-3);
///
/// // Bayesian conjugate estimate
/// let mut b = Bernoulli::new(n, None).condition(Beta::new(1,1));
/// b.fit(y);
/// let post : Beta = b.take_factor().unwrap();
/// assert!(post.mean()[0] - 0.5 < 1E-3);
/// ```
#[derive(Debug)]
pub struct Bernoulli {

    theta : DVector<f64>,

    /// Log-prob argument when struct represent a conditional expectation.
    eta : DVector<f64>,

    factor : BernoulliFactor,

    eta_traj : Option<EtaTrajectory>,

    sampler : Vec<rand_distr::Bernoulli>,

    log_part : DVector<f64>,

    /// In case this has a beta factor, when updating the parameter, also update this buffer
    /// with [log(theta) log(1-theta)]; and pass it instead of the eta as the log-prob argument.
    suf_theta : Option<DMatrix<f64>>

}

impl Bernoulli {

    /// logit informs if the constructor parameter
    /// and any incoming set_parameter calls should be interpreted as the natural parameter.
    /// Bernoullis parametrized by the logit cannot have conjugate beta factors; but can
    /// be conditioned on generic unconstrained continuous parameters such as the ones
    /// generated by a normal distribution.
    pub fn new(n : usize, theta : Option<f64>) -> Self {
        let mut bern : Bernoulli = Default::default();
        let theta = DVector::from_element(n, theta.unwrap_or(0.5));
        bern.set_parameter(theta.rows(0,theta.nrows()), false);
        bern
    }

    pub fn factorial(n: usize) -> usize {
        if n < 2 {
            1
        } else {
            n * Self::factorial(n - 1)
        }
    }

}

impl Conditional<Beta> for Bernoulli {

    fn condition(mut self, b : Beta) -> Bernoulli {
        self.factor = BernoulliFactor::Conjugate(b);
        self.suf_theta = Some(Beta::sufficient_stat(self.theta.slice((0, 0), (self.theta.nrows(), 1))));
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
        self.log_part.iter_mut().zip(eta.iter()).for_each(|(l,e)| { *l = (1. + e.exp()).ln(); } );
    }

    fn log_partition<'a>(&'a self) -> &'a DVector<f64> {
        &self.log_part
    }

    fn update_grad(&mut self, _eta : DVectorSlice<'_, f64>) {
        unimplemented!()
    }

    fn grad(&self) -> &DVector<f64> {
        unimplemented!()
    }

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

impl Likelihood<U1> for Bernoulli {

    fn mean_mle(y : DMatrixSlice<'_, f64>) -> f64 {
        assert!(y.ncols() == 1);
        let mle = y.iter().fold(0.0, |ys, y| {
            assert!(*y == 0. || *y == 1.); ys + y
        }) / (y.nrows() as f64);
        mle
    }

    fn var_mle(y : DMatrixSlice<'_, f64>) -> f64 {
        let m = Self::mean_mle(y);
        m * (1. - m)
    }

}

impl Estimator<Beta> for Bernoulli {

    fn fit<'a>(&'a mut self, y : DMatrix<f64>) -> Result<&'a Beta, &'static str> {
        assert!(y.ncols() == 1);
        match self.factor {
            BernoulliFactor::Conjugate(ref mut beta) => {
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
    }

    fn view_parameter(&self, natural : bool) -> &DVector<f64> {
        match natural {
            true => &self.eta,
            false => &self.theta
        }
    }

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

    fn log_prob(&self, y : DMatrixSlice<f64>) -> f64 {
        assert!(y.ncols() == 1);
        let eta = match self.current() {
            Some(eta) => eta,
            None => self.eta.rows(0, self.eta.nrows())
        };
        let factor_lp = match &self.factor {
            BernoulliFactor::Conjugate(b) => {
                b.log_prob(self.suf_theta.as_ref().unwrap().slice((0,0), (1,2)))
            },
            BernoulliFactor::CondExpect(m) => {
                m.suf_log_prob(eta.slice((0,0), (eta.nrows(), 1)))
            },
            BernoulliFactor::Empty => 0.
        };
        eta.dot(&y) - self.log_part[0] + factor_lp
    }

    fn sample(&self) -> DMatrix<f64> {
        use rand_distr::{Distribution};
        let mut samples = DMatrix::zeros(self.theta.nrows(), 1);
        for (i, _) in self.theta.iter().enumerate() {
            samples[(i,1)] = (self.sampler[i].sample(&mut rand::thread_rng()) as i32) as f64;
        }
        samples
    }

}

impl RandomWalk for Bernoulli {

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

}

impl Default for Bernoulli {

    fn default() -> Self {
        Bernoulli {
            theta : DVector::from_element(1, 0.5),
            eta : DVector::from_element(1, 0.0),
            factor : BernoulliFactor::Empty,
            eta_traj : None,
            sampler : Vec::new(),
            log_part : DVector::from_element(1, (2.).ln()),
            suf_theta : None,
        }
    }

}


