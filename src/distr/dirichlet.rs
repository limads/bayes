use nalgebra::*;
// use rand_distr;
use super::*;
use serde::{Serialize, Deserialize};
use std::fmt::{self, Display};
use crate::inference::sim::RandomWalk;
use super::MultiNormal;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dirichlet {

    rw : Option<RandomWalk>,

    approx : Option<MultiNormal>
}

impl Distribution for Dirichlet {

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

    fn sample_into(&self, _dst : DMatrixSliceMut<'_, f64>) {
        // let dirichlet = Dirichlet::new(vec![1.0, 2.0, 3.0]).unwrap();
        // let samples = dirichlet.sample(&mut rand::thread_rng());*/
        unimplemented!()
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        None
    }

    fn cov_inv(&self) -> Option<DMatrix<f64>> {
        None
    }


}

impl Posterior for Dirichlet {

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Posterior>, Option<&mut dyn Posterior>) {
        unimplemented!()
    }

    fn approximation_mut(&mut self) -> Option<&mut MultiNormal> {
        self.approx.as_mut()
    }

    fn approximation(&self) -> Option<&MultiNormal> {
        self.approx.as_ref()
    }

    fn trajectory(&self) -> Option<&RandomWalk> {
        self.rw.as_ref()
    }

    fn trajectory_mut(&mut self) -> Option<&mut RandomWalk> {
        self.rw.as_mut()
    }

}

impl ExponentialFamily<Dynamic> for Dirichlet
    where
        Self : Distribution
{

    fn base_measure(_y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        unimplemented!()
    }

    fn sufficient_stat(_y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        unimplemented!()
    }

    fn suf_log_prob(&self, _t : DMatrixSlice<'_, f64>) -> f64 {
        unimplemented!()
    }

    fn update_log_partition<'a>(&'a mut self, _eta : DVectorSlice<'_, f64>) {
        unimplemented!()
    }

    fn log_partition<'a>(&'a self) -> &'a DVector<f64> {
        unimplemented!()
    }

    /*fn update_grad(&mut self, _eta : DVectorSlice<'_, f64>) {
        unimplemented!()
    }

    fn grad(&self) -> &DVector<f64> {
        unimplemented!()
    }*/

    fn link_inverse<S>(_eta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        unimplemented!()
    }

    fn link<S>(_theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        unimplemented!()
    }

}

impl Display for Dirichlet {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dir(1)")
    }

}

