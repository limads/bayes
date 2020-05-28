use nalgebra::*;
use super::*;
use crate::gsl::gamma::*;
use rand_distr; //::{ /*Distribution,*/ Gamma};
use rand;
use serde::{Serialize, Deserialize};
use serde::ser::{Serializer};
use serde::de::Deserializer;

/// Gamma is a distribution for scale parameters. For a location parameter centered
/// at alpha (shape), Gamma(alpha, beta) represents the random distribution of
/// a relative change alpha*(1/beta) for this parameter. For alpha=1, the gamma distribution has its mode truncated
/// at zero (which gives rise to the exponential distribution with shape beta; or the distribution of
/// the squared deviation of a single standard normal (X^2(1)). A sample of size n from the standard normal
/// has its precision distributed as Gamma with alpha=n/2. For alpha >> 1, the gamma approaches
/// a gaussian centered at alpha / beta and with dispersion alpha / beta^2.
/// TODO make alpha fixed at construction; so to characterize a pseudo-sample of size n, set alpha=n/2.
/// In this way, Gamma implements univariate distribution. To use as exponential prior, set alpha=1.
#[derive(Debug, Clone)]
pub struct Gamma {

    //joint_ix : (usize, usize),

    /// Size-2 parameter vector with shape alpha and inverse-scale beta.
    /// For fixed beta, Gamma is closed under addition.
    ab : DVector<f64>,

    eta : DVector<f64>,

    mean : DVector<f64>,

    log_part : DVector<f64>,

    sampler : rand_distr::Gamma<f64>
}

impl Gamma {

    pub fn new(alpha : f64, beta : f64) -> Self {
        let mut gamma : Gamma = Default::default();
        let ab = DVector::from_column_slice(&[alpha, beta]);
        gamma.set_parameter(ab.rows(0,2), false);
        gamma
    }

    pub fn gamma(y : f64) -> f64 {
        unsafe{ gsl_sf_gamma(y) }
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

    /*fn sufficient_stat(y : DMatrix<f64>) -> DMatrix<f64> {
        if y.ncols() > 1 {
            panic!("The Gamma distribution can only be evaluated at a single column");
        }
        let y0 = y.column(0).map(|y| y.ln());
        let y1 = y.column(0).map(|y| y);
        DMatrix::from_columns(&[y0, y1])
    }*/

    /// y: Vector of precision draws
    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        assert!(y.ncols() == 1);
        let mut suf = DMatrix::zeros(2, 1);
        for y in y.column(0).iter() {
            suf[(0,0)] += y.ln();
            suf[(1,0)] += y;
        }
        suf
    }

    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {
        assert!(self.log_part.nrows() == 1);
        assert!(t.ncols() == 1 && t.nrows() == 2);
        self.eta.dot(&t.column(0)) - self.log_part[0]
    }

    fn update_log_partition<'a>(&'a mut self, eta : DVectorSlice<'_, f64>) {
        let log_part_v = Gamma::gamma(eta[0] + 1.).ln() -
            (eta[0] + 1.)*(-eta[1]).ln();
        self.log_part = DVector::from_element(1, log_part_v);
    }

    fn log_partition<'a>(&'a self) -> &'a DVector<f64> {
        &self.log_part
    }

    fn update_grad(&mut self, eta : DVectorSlice<'_, f64>) {
        unimplemented!()
    }

    fn grad(&self) -> &DVector<f64> {
        unimplemented!()
    }

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
        let ab = Self::link_inverse(&eta);
        self.mean = DVector::from_element(1, ab[0] / ab[1]);
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        None
    }


    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.mean
    }

    fn mode(&self) -> DVector<f64> {
        /// For alpha (shape) <= 1, the mode of the beta is truncated at zero.
        /// For alpha >> 1, the mode approaces the mode of the gaussian at alpha / beta.
        DVector::from_element(1, (self.ab[0] - 1.) / self.ab[1])
    }

    fn var(&self) -> DVector<f64> {
        DVector::from_element(1, self.ab[0] / self.ab[1].powf(2.))
    }

    fn log_prob(&self, y : DMatrixSlice<f64>) -> f64 {
        let suf = Self::sufficient_stat(y);
        self.suf_log_prob(suf.rows(0,suf.nrows()))
    }

    //fn shared_factors(&self) -> Factors {
    //    Vec::new().into()
    //}

    //fn cond_log_prob(&self, y : DMatrixSlice<f64>, joint : &DVector<f64>) -> f64 {
    //    unimplemented!()
    //}

    fn sample(&self) -> DMatrix<f64> {
        use rand_distr::Distribution;
        let s = self.sampler.sample(&mut rand::thread_rng());
        DMatrix::from_element(1, 1, s)
    }

    /*fn factors<'b>(&'b self) -> Factors<'b> {
        unimplemented!()
    }

    fn factors_mut<'b>(&'b mut self) -> FactorsMut<'b> {
        unimplemented!()
    }

    fn marginalize(&self) -> Option<Histogram> {
        None
    }*/

    // fn shift_ix(&mut self, by : usize) {
    //    self.joint_ix.0 += by
    // }

    //fn retrieve_parameter<'a>(&'a self, joint : &'a DVector<f64>) -> Option<DVectorSlice<'a, f64>> {
    //    Some(joint.rows(self.joint_ix.0, self.joint_ix.1))
    //}

}

impl Serialize for Gamma {

    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        unimplemented!()
    }
}

impl<'de> Deserialize<'de> for Gamma {

    fn deserialize<D>(deserializer: D) -> Result<Gamma, D::Error>
    where
        D: Deserializer<'de>,
    {
        unimplemented!()
    }
}

pub struct ChiSquare {
    g : Gamma
}

impl ChiSquare {

    pub fn new(p : usize) -> Self {
        ChiSquare{ g : Gamma::new(p as f64 / 2., 0.5 ) }
    }

}

pub struct Exponential {
    g : Gamma
}

impl Exponential {

    pub fn new(lambda : f64) -> Self {
        Exponential { g : Gamma::new(1.0, lambda) }
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
            sampler : rand_distr::Gamma::new(1., 1.).unwrap()
        }
    }

}

