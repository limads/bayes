use nalgebra::*;
use super::*;
use serde::{Serialize, Deserialize};
use std::fmt::{self, Display};
use crate::fit::markov::Trajectory;
use super::MultiNormal;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dirichlet {

    traj : Option<Trajectory>,

    approx : Option<MultiNormal>
}

impl Distribution for Dirichlet {

    fn set_parameter(&mut self, _p : DVectorSlice<'_, f64>, _natural : bool) {
        unimplemented!()
    }

    fn view_parameter(&self, D_natural : bool) -> &DVector<f64> {
        unimplemented!()
    }

    fn set_natural<'a>(&'a mut self, eta : &'a mut dyn Iterator<Item=&'a f64>) {
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

    fn joint_log_prob(&self, /*_y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>*/ ) -> Option<f64> {
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

impl Markov for Dirichlet {

    fn natural_mut<'a>(&'a mut self) -> DVectorSliceMut<'a, f64> {
        unimplemented!()
    }

    fn canonical_mut<'a>(&'a mut self) -> Option<DVectorSliceMut<'a, f64>> {
        unimplemented!()
    }

}

impl Posterior for Dirichlet {

    /*fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Posterior>, Option<&mut dyn Posterior>) {
        unimplemented!()
    }*/

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

    fn update_log_partition<'a>(&'a mut self, /*_eta : DVectorSlice<'_, f64>*/ ) {
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

