use nalgebra::*;
use crate::distr::*;
// use super::*;
// use crate::optim::*;
use std::ops::AddAssign;
use serde::{Serialize, Deserialize};
use std::ops::Index;

/// Structure to represent one-dimensional empirical distributions non-parametrically (Work in progress).
pub mod histogram;

pub use histogram::*;

/// Metropolis-Hastings posterior sampler (Work in progress).
pub mod metropolis;

pub use metropolis::*;

/// A sequence of natural parameter iterations. The distribution at the current node
/// holds the parameter trajectory of all distributions at the parent nodes, which during
/// optimization or posterior sampling are considered as conditioned or unconditional priors.
/// After optimization/simulation, this trajectory is used to build an approximation to the
/// corresponding posterior entry, which can be retrieved via node.approximate() or node.marginal().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EtaTrajectory {

    pub pos : usize,

    pub traj : DMatrix<f64>,

    pub weights : DVector<f64>

}

impl EtaTrajectory {

    pub fn new(start : DVectorSlice<'_, f64>) -> Self {
        let mut traj = DMatrix::zeros(start.nrows(), 1000);
        let weights = DVector::from_element(1000, 1.);
        let pos = 0;
        traj.column_mut(0).copy_from(&start);
        Self{ pos, traj, weights }
    }

    pub fn step(&mut self, opt_data : Option<DVectorSlice<'_, f64>>) {
        if let Some(data) = opt_data {
            self.pos += 1;
            if self.traj.ncols() == self.pos {
                self.traj = self.traj.clone()
                    .insert_columns(self.traj.ncols(), 1000, 0.);
                self.weights = self.weights.clone()
                    .insert_rows(self.weights.nrows(), 1000, 1.);
            }
            self.traj.column_mut(self.pos).copy_from(&data);
        } else {
            self.weights[self.pos] += 1.;
        }
    }

    pub fn step_increment(&mut self, incr : DVectorSlice<'_, f64>) {
        let prev : DVector<f64> = self.traj.column(self.pos).into();
        self.traj.column_mut(self.pos+1).copy_from(&prev);
        self.traj.column_mut(self.pos+1).add_assign(&incr);
        self.pos += 1;
    }

    pub fn get<'a>(&'a self) -> DVectorSlice<'a, f64> {
        self.traj.column(self.pos)
    }
}

/// Sample is a collection of 1D posterior marginals, recovered via indexing.
pub struct Sample {

    /// Applies link function to eta_traj to get this field.
    _theta_traj : DMatrix<f64>

}

impl Sample {

    pub fn new(_theta_traj : DMatrix<f64>) -> Self {
        Self{ _theta_traj }
    }

}

impl Index<usize> for Sample {

    type Output = Histogram;

    fn index(&self, _ix: usize) -> &Self::Output {
        unimplemented!()
    }

}

/// RandomWalk is implemented by distributions who may
/// maintain a history of changes in the natural parameter
/// scale of its parent(s) node(s).
pub trait RandomWalk
    where
        Self : Distribution
{

    /// Returns the current state (eta).
    fn current<'a>(&'a self) -> Option<DVectorSlice<'a, f64>>;

    /// After each increment, the implementor should have its
    /// log-probability and gradient evaluated with respect to (eta_t-1 + eta_diff)
    /// but its sampling state and statistics are still defined by the last
    /// saved state. Update informs whether the internal distribution state should
    /// be updated given the step. If not set, only log_prob(.) output will be affected.
    fn step_by<'a>(&'a mut self, diff_eta : DVectorSlice<'a, f64>, update : bool);

    /// Update internal natural parameter to the informed value.
    /// If there is not an new_eta
    /// data vector, just increment the previous value by one.
    fn step_to<'a>(&'a mut self, new_eta : Option<DVectorSlice<'a, f64>>, update : bool);

    /// Use the implementor trajectory as a non-parametric representation
    /// of a marginal probability distribution. Applies any necessary transformations to
    /// the eta trajectory so that the trajectory is now represented with respect to theta.
    fn marginal(&self) -> Option<Sample>;

    /*/// Runtime description for the implementor.
    fn description() -> SourceDistribution;*/

}


