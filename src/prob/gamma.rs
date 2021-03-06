use nalgebra::*;
use super::*;
use rand_distr; //::{ /*Distribution,*/ Gamma};
use rand;
use serde::{Serialize, Deserialize};
use serde::ser::{Serializer};
use serde::de::Deserializer;
use std::fmt::{self, Display};
use crate::fit::markov::Trajectory;
use mathru::special;

#[cfg(feature="gsl")]
use crate::foreign::gsl::gamma::*;

/// The Gamma is a distribution for inverse-scale or rate parameters. For a location parameter centered
/// at alpha (shape), Gamma(alpha, beta) represents the random distribution of
/// a relative change alpha*(1/beta) for this parameter. For alpha=1, the gamma distribution has its mode truncated
/// at zero (which gives rise to the exponential distribution with shape beta; or the distribution of
/// the squared deviation of a single standard normal (X^2(1)). A sample of size n from the standard normal
/// has its precision distributed as Gamma with alpha=n/2. For alpha >> 1, the gamma approaches
/// a gaussian centered at alpha / beta and with dispersion alpha / beta^2.
/// For a fixed alpha=1, Gamma behaves as an exponential distribution, taking the sum of interval samples as its sufficient statistic.
/// A Gamma distribution can be seen as a closed-form, finite-sample, slightly biased distribution
/// for the estimate of an inverse shape (or rate) parameter, where the bias is given by the size of the pseudo-sample considered.
/// When a is fixed at any positive integer, we have the Erlang distribution.
#[derive(Debug, Clone)]
pub struct Gamma {

    //joint_ix : (usize, usize),

    /// Size-2 parameter vector with shape alpha and inverse-scale beta.
    /// For fixed beta, Gamma is closed under addition.
    ab : DVector<f64>,

    eta : DVector<f64>,

    mean : DVector<f64>,

    log_part : DVector<f64>,

    sampler : rand_distr::Gamma<f64>,

    factor : Option<Box<Gamma>>,

    traj : Option<Trajectory>,

    approx : Option<MultiNormal>
}

impl Gamma {

    pub fn new(alpha : f64, beta : f64) -> Self {
        assert!(alpha > 0.0, "alpha should be greater than zero");
        assert!(beta > 0.0, "beta should be greater than zero");
        let mut gamma : Gamma = Default::default();
        let ab = DVector::from_column_slice(&[alpha, beta]);
        gamma.set_parameter(ab.rows(0,2), false);
        gamma
    }

    pub fn gamma(y : f64) -> f64 {
        #[cfg(feature="gsl")]
        return unsafe{ gsl_sf_gamma(y) };
        
        special::gamma::gamma(y)
    }

    pub fn ln_gamma(y : f64) -> f64 {
        #[cfg(feature="gsl")]
        return unsafe{ gsl_sf_lngamma(y) };
        
        special::gamma::ln_gamma(y)
    }

    pub fn gamma_inv(y : f64) -> f64 {
        #[cfg(feature="gsl")]
        return unsafe{ gsl_sf_gammainv(y) };
        
        panic!("Use of the gamma/beta prior require gsl feature");
    }

}

impl ExponentialFamily<Dynamic> for Gamma {

    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64>
        //where S : Storage<f64, Dynamic, Dynamic>
    {
        if y.ncols() > 2 {
            panic!("The Gamma distribution can only be evaluated at a single data point");
        }
        DVector::from_element(1, 1.)
    }

    /// y: Vector of precision/rate draws
    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        assert!(y.ncols() == 1, "Sample should have single column");
        let mut suf = DMatrix::zeros(2, 1);
        for y in y.column(0).iter() {
            assert!(*y > 0.0, "Gamma should be evaluated against strictly positive values.");
            suf[(0,0)] += y.ln();
            suf[(1,0)] += y;
        }
        suf
    }

    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {
        assert!(self.log_part.nrows() == 1, "Log-partition 1x1");
        assert!(t.ncols() == 1 && t.nrows() == 2, "Sufficient statistic matrix should be 2x1");
        self.eta.dot(&t.column(0)) - self.log_part[0]
    }

    fn update_log_partition<'a>(&'a mut self, /*eta : DVectorSlice<'_, f64>*/ ) {
        let log_part_v = Gamma::ln_gamma(self.eta[0] + 1.) - (self.eta[0] + 1.)*(-1.*self.eta[1]).ln();
        self.log_part = DVector::from_element(1, log_part_v);
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
        let theta_0 = eta[0] + 1.;
        let theta_1 = (-1.)*eta[1];
        DVector::from_column_slice(&[theta_0, theta_1])
    }

    fn link<S>(theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        let eta_0 = theta[0] - 1.;
        let eta_1 = (-1.)*theta[1];
        DVector::from_column_slice(&[eta_0, eta_1])
    }
}

impl Distribution for Gamma
    where Self : Sized
{

    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, natural : bool) {
        let eta = match natural {
            true => p.clone_owned(),
            false => Self::link(&p)
        };
        self.set_natural(&mut eta.iter());
    }

    fn dyn_factors(&self) -> (Option<&dyn Distribution>, Option<&dyn Distribution>) {
        match &self.factor {
            Some(ref g) => (Some(g.as_ref() as &dyn Distribution), None),
            None => (None, None)
        }
    }

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Distribution>, Option<&mut dyn Distribution>) {
        match &mut self.factor {
            Some(ref mut g) => (Some(g.as_mut() as &mut dyn Distribution), None),
            None => (None, None)
        }
    }

    fn set_natural<'a>(&'a mut self, new_eta : &'a mut dyn Iterator<Item=&'a f64>) {
        let eta = DVector::from_iterator(2, new_eta.cloned());
        let ab = Self::link_inverse(&eta);
        self.mean = DVector::from_element(1, ab[0] / ab[1]);
        self.eta = eta;
        self.update_log_partition();
    }

    fn view_parameter(&self, natural : bool) -> &DVector<f64> {
        match natural {
            true => &self.eta,
            false => &self.ab
        }
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        None
    }

    fn cov_inv(&self) -> Option<DMatrix<f64>> {
        None
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.mean
    }

    fn mode(&self) -> DVector<f64> {
        // For alpha (shape) <= 1, the mode of the beta is truncated at zero.
        // For alpha >> 1, the mode approaces the mode of the gaussian at alpha / beta.
        DVector::from_element(1, (self.ab[0] - 1.) / self.ab[1])
    }

    fn var(&self) -> DVector<f64> {
        DVector::from_element(1, self.ab[0] / self.ab[1].powf(2.))
    }

    fn joint_log_prob(&self /*, y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>*/ ) -> Option<f64> {
        /*assert!(y.ncols() == 1, "Gamma sample should have single column");
        let suf = Self::sufficient_stat(y);
        self.suf_log_prob(suf.rows(0,suf.nrows()))*/
        unimplemented!()
    }

    fn sample_into(&self, mut dst : DMatrixSliceMut<'_,f64>) {
        use rand_distr::Distribution;
        let g = self.sampler.sample(&mut rand::thread_rng());
        dst[(0,0)] = g;
    }

}

impl Posterior for Gamma {

    fn approximation_mut(&mut self) -> Option<&mut MultiNormal> {
        self.approx.as_mut()
    }

    fn approximation(&self) -> Option<&MultiNormal> {
        self.approx.as_ref()
    }

    fn trajectory(&self) -> Option<&Trajectory> {
        self.traj.as_ref()
    }

    fn trajectory_mut(&mut self) -> Option<&mut Trajectory> {
        self.traj.as_mut()
    }
    
    fn start_trajectory(&mut self, size : usize) {
        self.traj = Some(Trajectory::new(size, self.view_parameter(true).nrows()));
    }
    
    /// Finish the trajectory before its predicted end.
    fn finish_trajectory(&mut self) {
        self.traj.as_mut().unwrap().closed = true;
    }

}

impl Markov for Gamma {

    fn natural_mut<'a>(&'a mut self) -> DVectorSliceMut<'a, f64> {
        self.eta.column_mut(0)
    }

    fn canonical_mut<'a>(&'a mut self) -> Option<DVectorSliceMut<'a, f64>> {
        None
    }

}

impl Serialize for Gamma {

    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        unimplemented!()
    }
}

impl<'de> Deserialize<'de> for Gamma {

    fn deserialize<D>(_deserializer: D) -> Result<Gamma, D::Error>
    where
        D: Deserializer<'de>,
    {
        unimplemented!()
    }
}

impl Default for Gamma {

    fn default() -> Self {
        let ab = DVector::from_column_slice(&[1., 1.]);
        Self {
            eta : Self::link(&ab),
            ab : ab,
            mean : DVector::from_element(1, 0.5),
            log_part : DVector::from_element(1, 0.0),
            sampler : rand_distr::Gamma::new(1., 1.).unwrap(),
            factor : None,
            approx : None,
            traj : None
        }
    }

}

impl Display for Gamma {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gam(1)")
    }

}

