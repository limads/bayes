use super::*;
// use serde::{Serialize, Deserialize};
use super::gamma::*;
// use nalgebra::*;
// use nalgebra::storage::*;
use rand_distr;
use rand;
use std::default::Default;

/// A beta distribution yields ratios over the interval [0, 1], produced by taking the
/// ratio of two independent gamma distributions: If u ~ Gamma(n/2, 0.5) and v ~ Gamma(m/2, 0.5)
/// Then u / (u + v) ~ Beta. It is commonly used to model prior Bernoulli probabilities, where
/// m and n are the pseudo-data dimensionality of a success trial count and its complement.
/// By setting u = v >> 0, Beta approaches a gaussian distribution centered at u / (u + v). By setting
/// u = v = 1, Beta equals a uniform distribution bounded at [0,1]. Alternatively, a Beta can be seen
/// as a closed-form, finite-sample, slightly biased distribution for the estimate of a proportion parameter,
/// where the bias is given by the size of the pseudo-sample considered.
#[derive(Debug, Clone)]
pub struct Beta {

    // For a full joint evaluation, what its the first index and the lenght of the parameters
    // that correspond to this distribution.
    // joint_ix : (usize, usize),

    // Represent vector of a; vector of b.
    ab : DVector<f64>,

    mean : DVector<f64>,

    sampler : rand_distr::Beta<f64>,

    log_part : DVector<f64>,

    // factor : Option<Box<Beta>>

}

// use rand_distr::{Distribution, Beta};
// let beta = Beta::new(2.0, 5.0).unwrap();
// let v = beta.sample(&mut rand::thread_rng());
// println!("{} is from a Beta(2, 5) distribution", v);

impl Beta {

    pub fn new(a : usize, b : usize) -> Self {
        let mut beta : Beta = Default::default();
        let ab = DVector::from_column_slice(&[a as f64, b as f64]);
        beta.set_parameter(ab.rows(0, 2), false);
        beta
    }

}

impl ExponentialFamily<Dynamic> for Beta {

    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        println!("y={}", y);
        if y.ncols() > 2 {
            panic!("The Beta distribution can only be evaluated at a single data point");
        }
        let theta = y[0];
        DVector::from_element(1, 1. / (theta * (1. - theta)) )
    }

    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        assert!(y.ncols() == 1);
        let mut suf = DMatrix::zeros(2, 1);
        for y in y.column(0).iter() {
            suf[(0,0)] += (y + 1E-10).ln();
            suf[(1,0)] += (1. - y).ln()
        }
        suf
    }

    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {
        assert!(t.ncols() == 1 && t.nrows() == 2);
        assert!(self.log_part.nrows() == 1);
        self.ab.dot(&t.column(0)) - self.log_part[0]
    }

    fn update_log_partition<'a>(&'a mut self, eta : DVectorSlice<'_, f64>) {
        println!("{}", eta);
        let log_part_val = Gamma::ln_gamma(eta[0] as f64) +
            Gamma::ln_gamma(eta[1] as f64) -
            Gamma::ln_gamma(eta[0] as f64 + eta[1] as f64 );
        self.log_part = DVector::from_element(1, log_part_val);
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
        DVector::from_iterator(eta.nrows(), eta.iter().map(|t| *t))
    }

    fn link<S>(theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
       DVector::from_iterator(theta.nrows(), theta.iter().map(|t| *t))
    }
}

impl Distribution for Beta
    where Self : Sized
{

    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, _natural : bool) {
        self.ab = p.clone_owned();
        let (a, b) = (p[0], p[1]);
        self.mean = DVector::from_element(1, a / (a + b));
        self.update_log_partition(p);
    }

    fn view_parameter(&self, _natural : bool) -> &DVector<f64> {
        &self.ab
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        None
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.mean
    }

    fn mode(&self) -> DVector<f64> {
        let (a, b) = (self.ab[0], self.ab[1]);
        DVector::from_column_slice(&[a - 1., a + b - 2.])
    }

    fn var(&self) -> DVector<f64> {
        let (a, b) = (self.ab[0], self.ab[1]);
        DVector::from_element(1, a*b / (a + b).powf(2.) * (a + b + 1.))
    }

    fn log_prob(&self, y : DMatrixSlice<f64>) -> f64 {
        assert!(y.ncols() == 1);
        let t = Beta::sufficient_stat(y);
        self.suf_log_prob((&t).into())
    }

    /*fn shared_factors(&self) -> Factors {
        Vec::new().into()
    }

    /*fn cond_log_prob(&self, y : DMatrixSlice<f64>, joint : &DVector<f64>) -> f64 {
        unimplemented!()
    }*/

    /*fn update(&mut self, joint : &DVector<f64>) {
        unimplemented!()
    }*/*/

    fn sample(&self) -> DMatrix<f64> {
        use rand_distr::Distribution;
        let b = self.sampler.sample(&mut rand::thread_rng());
        DMatrix::from_element(1, 1, b)
    }

    /*fn factors<'b>(&'b self) -> Factors<'b> {
        unimplemented!()
    }

    /// Iterate over mutable references of the nodes of this
    /// distribution, so they can be conditioned at other values. If this distribution
    /// is a root node, this iterator yield no values.
    fn factors_mut<'b>(&'b mut self) -> FactorsMut<'b> {
        unimplemented!()
    }*/

    /*fn marginalize(&self) -> Option<Histogram> {
        None
    }*/

    // fn retrieve_parameter<'a>(&'a self, joint : &'a DVector<f64>) -> Option<DVectorSlice<'a, f64>> {
        //Some(joint.rows(self.joint_ix.0, self.joint_ix.1))
    //    unimplemented!()
    // }

}

impl Default for Beta {

    fn default() -> Self {
        Self {
            ab : DVector::from_column_slice(&[1., 1.]),
            mean : DVector::from_element(1, 0.5),
            log_part : DVector::from_element(1, 0.0),
            sampler : rand_distr::Beta::new(1., 1.).unwrap()
        }
    }

}

/*impl ConditionalDistribution<BinaryOp<F,G>> for Beta {
    unimplemented!()
}*/
