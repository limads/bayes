use super::*;
use super::gamma::*;
use serde::{Serialize, Deserialize};
use rand_distr;
// use rand;
use crate::sim::*;
// use std::ops::AddAssign;
use std::default::Default;
use super::bernoulli::*;
use serde::ser::{Serializer};
use serde::de::Deserializer;

pub type PoissonFactor = UnivariateFactor<Gamma>;

#[derive(Debug, Clone)]
pub struct Poisson {

    lambda : DVector<f64>,

    eta : DVector<f64>,

    factor : PoissonFactor,

    log_part : DVector<f64>,

    eta_traj : Option<EtaTrajectory>,

    sampler : Vec<rand_distr::Poisson<f64>>,

    suf_lambda : Option<DMatrix<f64>>

}

impl Poisson {

    pub fn new(n : usize, lambda : Option<f64>) -> Self {
        let mut p : Poisson = Default::default();
        let l = DVector::from_element(n, lambda.unwrap_or(1.));
        p.set_parameter(l.rows(0, l.nrows()), false);
        p
    }
}

impl ExponentialFamily<U1> for Poisson
    where
        Self : Distribution
{

    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        assert!(y.ncols() == 1);
        y.column(0).map(|y| 1. / (Bernoulli::factorial(y as usize) as f64))
    }

    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        assert!(y.ncols() == 1);
        DMatrix::from_element(1, 1, y.column(0).iter().fold(0.0, |ys, y| ys + y ))
    }

    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {
        assert!(t.nrows() == 1 && t.ncols() == 1);
        assert!(self.eta.nrows() == 1);
        assert!(self.log_part.nrows() == 1);
        self.eta[0] * t[0] - self.log_part[0]
    }

    fn update_log_partition<'a>(&'a mut self, eta : DVectorSlice<'_, f64>) {
        self.log_part.iter_mut().zip(eta.iter()).for_each(|(l,e)| { *l = e.exp() } );
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
        eta.map(|e| e.exp() )
    }

    fn link<S>(theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        theta.map(|p| p.ln() )
    }

}

impl Distribution for Poisson
    where Self : Sized
{

    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, natural : bool) {
        let eta = if natural {
            p.clone_owned()
        } else {
            Self::link(&p)
        };
        self.lambda = Self::link_inverse(&eta);
        self.update_log_partition(eta.rows(0,eta.nrows()));
        self.eta = eta;
        if let Some(ref mut suf) = self.suf_lambda {
            *suf = Gamma::sufficient_stat(self.lambda.slice((0,0), (self.lambda.nrows(),1)));
        }
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.lambda
    }

    fn mode(&self) -> DVector<f64> {
        self.lambda.clone()
    }

    fn var(&self) -> DVector<f64> {
        self.lambda.clone()
    }

    fn log_prob(&self, y : DMatrixSlice<f64>) -> f64 {
        assert!(y.ncols() == 1);
        let eta = match self.current() {
            Some(eta) => eta,
            None => self.eta.rows(0, self.eta.nrows())
        };
        let factor_lp = match &self.factor {
            PoissonFactor::Conjugate(g) => {
                g.log_prob(self.suf_lambda.as_ref().unwrap().slice((0,0), (1,2)))
            },
            PoissonFactor::CondExpect(m) => {
                m.suf_log_prob(eta.slice((0,0), (eta.nrows(), 1)))
            },
            PoissonFactor::Empty => 0.
        };
        eta.dot(&y) - self.log_part[0] + factor_lp
    }

    fn sample(&self) -> DMatrix<f64> {
        /*use rand_distr::{Distribution};
        let mut samples = DMatrix::zeros(self.lambda.nrows(), 1);
        for (i, t) in self.lambda.iter().enumerate() {
            samples[(i,1)] = (self.sampler[i].sample(&mut rand::thread_rng()) as i32) as f64;
        }
        samples*/
        unimplemented!()
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        None
    }


}

impl RandomWalk for Poisson {

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

impl Default for Poisson {

    fn default() -> Self {
        Poisson {
            lambda : DVector::from_element(1, 0.5),
            eta : DVector::from_element(1, 0.0),
            factor : PoissonFactor::Empty,
            eta_traj : None,
            sampler : Vec::new(),
            suf_lambda : None,
            log_part : DVector::from_element(1, (2.).ln()),
        }
    }

}

impl Serialize for Poisson {

    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        unimplemented!()
    }
}

impl<'de> Deserialize<'de> for Poisson {

    fn deserialize<D>(_deserializer: D) -> Result<Poisson, D::Error>
    where
        D: Deserializer<'de>,
    {
        unimplemented!()
    }
}

