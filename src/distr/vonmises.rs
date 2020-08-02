use nalgebra::*;
use super::*;
// use std::fmt::{self, Display};
use serde::{Serialize, Deserialize};
// use super::Gamma;
use std::fmt::{self, Display};
use crate::sim::RandomWalk;
use super::MultiNormal;

/// Exponential-family distribution defined over -π ≥ θ ≥ π, resulting
/// from the observation of a periodic process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VonMises {
    approx : Option<Box<MultiNormal>>,
    rw : Option<RandomWalk>
}

impl Distribution for VonMises
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

impl Posterior for VonMises {

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Posterior>, Option<&mut dyn Posterior>) {
        unimplemented!()
    }

    fn approximation_mut(&mut self) -> Option<&mut MultiNormal> {
        self.approx.as_mut().map(|apprx| apprx.as_mut())
    }

    fn approximation(&self) -> Option<&MultiNormal> {
        self.approx.as_ref().map(|apprx| apprx.as_ref())
    }

    fn trajectory(&self) -> Option<&RandomWalk> {
        self.rw.as_ref()
    }

    fn trajectory_mut(&mut self) -> Option<&mut RandomWalk> {
        self.rw.as_mut()
    }

}

impl ExponentialFamily<U1> for VonMises
    where
        Self : Distribution
{

    fn base_measure(_y : DMatrixSlice<'_, f64>) -> DVector<f64>
        //where S : Storage<f64, Dynamic, U1>
    {
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

    /*n update_grad(&mut self, _eta : DVectorSlice<'_, f64>) {
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

impl Display for VonMises {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VMis(1)")
    }

}

