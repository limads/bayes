use nalgebra::*;
use super::*;
// use std::fmt::{self, Display};
use serde::{Serialize, Deserialize};
// use super::Gamma;

/// Exponential-family distribution defined over theta = [-pi,pi], resulting
/// from the observation of a periodic process. Useful to model the
/// common correlation component r of covariance matrices, since theta = acos(r) can be
/// seen as a draw from a VonMisses distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VonMises {

}

impl Distribution for VonMises
    where Self : Sized
{

    fn set_parameter(&mut self, _p : DVectorSlice<'_, f64>, _natural : bool) {
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

    fn log_prob(&self, _y : DMatrixSlice<f64>) -> f64 {
        unimplemented!()
    }

    fn sample(&self) -> DMatrix<f64> {
        unimplemented!()
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        None
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

    fn update_grad(&mut self, _eta : DVectorSlice<'_, f64>) {
        unimplemented!()
    }

    fn grad(&self) -> &DVector<f64> {
        unimplemented!()
    }

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
