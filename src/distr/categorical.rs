use nalgebra::*;
use super::*;
use super::dirichlet::*;
// use std::fmt::{self, Display};
use serde::{Serialize, Deserialize};
use std::fmt::{self, Display};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Categorical {
    theta : DVector<f64>,
    log_theta : DVector<f64>,
    eta : DVector<f64>,

    /// Unlike the univariate distributions, that hold a log-partition vector with same dimensionality
    /// as the parameter vector, the categorical holds a parameter vector with a single value, representing
    /// the state of the theta vector.
    log_part : DVector<f64>,

    factor : Option<Dirichlet>
}

impl Categorical {

    pub fn new(n : usize, param : Option<&[f64]>) -> Self {
        //println!("{}", param);
        //println!("{}", param.sum() + (1. - param.sum()) );
        let param = match param {
            Some(p) => {
                assert!(p.len() == n);
                DVector::from_column_slice(p)
            },
            None => DVector::from_element(n, 1. / ((n + 1) as f64))
        };
        assert!(param.sum() + (1. - param.sum()) - 1. < 10E-8);
        let eta = Self::link(&param);
        let mut cat = Self {
            log_theta : param.map(|t| t.ln() ),
            theta : param,
            eta : eta.clone(),
            log_part : DVector::from_element(1,1.),
            factor : None
        };
        cat.update_log_partition(eta.rows(0,eta.nrows()));
        cat
    }

}

impl Distribution for Categorical {

    fn view_parameter(&self, natural : bool) -> &DVector<f64> {
        match natural {
            true => &self.eta,
            false => &self.theta
        }
    }

    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, natural : bool) {
        if natural {
            self.eta = p.clone_owned();
            self.theta = Self::link_inverse(&self.eta);
        } else {
            self.theta = p.clone_owned();
            self.eta = Self::link(&self.theta);
        }
        self.log_theta = self.theta.map(|t| t.ln());
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.theta
    }

    fn mode(&self) -> DVector<f64> {
        self.theta.clone()
    }

    fn var(&self) -> DVector<f64> {
        self.theta.map(|theta| theta * (1. - theta))
    }

    fn log_prob(&self, y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>) -> f64 {
        let t  = Self::sufficient_stat(y.rows(0, y.nrows()));
        let factor_lp = match &self.factor {
            Some(dir) => {
                dir.suf_log_prob(self.log_theta.slice((0, 0), (self.log_theta.nrows(), 1)))
            },
            None => 0.0
        };
        self.suf_log_prob(t.rows(0, t.nrows())) + factor_lp
    }

    fn sample_into(&self, _dst : DMatrixSliceMut<'_, f64>) {
        unimplemented!()
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        let d = self.mean().nrows();
        let mut m = DMatrix::zeros(d, d);
        m.set_diagonal(&DVector::from_element(d, 1.));
        Some(m)
    }

    fn cov_inv(&self) -> Option<DMatrix<f64>> {
        self.cov()
    }

}

impl ExponentialFamily<Dynamic> for Categorical
    where
        Self : Distribution
{

    fn base_measure(_y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        DVector::from_element(1,1.)
    }

    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        DMatrix::from_column_slice(y.ncols(), 1, y.row_sum().as_slice())
    }

    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {
        assert!(t.ncols() == 1);
        self.eta.dot(&t.column(0)) - self.log_part[0]
    }

    fn update_log_partition<'a>(&'a mut self, eta : DVectorSlice<'_, f64>) {
        let e_sum = eta.iter().fold(0.0, |acc, e| acc + e.exp() );
        self.log_part[0] = (1. + e_sum).ln();
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
        let exp_eta = eta.map(|e| e.exp());
        let exp_sum = DVector::from_element(eta.nrows(), exp_eta.sum());
        let exp_sum_inv = exp_sum.map(|s| 1. / (1. + s));
        exp_sum_inv.component_mul(&exp_eta)
    }

    fn link<S>(theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        let prob_compl = 1. - theta.sum();
        theta.map(|t| (t / prob_compl).ln() )
    }

}

impl Conditional<Dirichlet> for Categorical {

    fn condition(self, _d : Dirichlet) -> Self {
        unimplemented!()
    }

    fn view_factor(&self) -> Option<&Dirichlet> {
        unimplemented!()
    }

    fn take_factor(self) -> Option<Dirichlet> {
        unimplemented!()
    }

    fn factor_mut(&mut self) -> Option<&mut Dirichlet> {
        unimplemented!()
    }

}

impl Display for Categorical {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cat({})", self.theta.nrows())
    }

}



