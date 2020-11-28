use nalgebra::*;
use crate::prob::*;
// use super::*;
// use crate::optim::*;
use std::ops::AddAssign;
use serde::{Serialize, Deserialize};
use std::ops::Index;
use nalgebra::storage::*;

/// Structure to represent one-dimensional empirical distributions non-parametrically (Work in progress).
pub mod histogram;

pub use histogram::*;

/// Metropolis-Hastings posterior sampler (Work in progress).
pub mod metropolis;

pub use metropolis::*;

pub mod marginal;

pub use marginal::*;

/// A sequence of natural parameter iterations. The distribution at the current node
/// holds the parameter trajectory of all distributions at the parent nodes, which during
/// optimization or posterior sampling are considered as conditioned or unconditional priors.
/// After optimization/simulation, this trajectory is used to build an approximation to the
/// corresponding posterior entry, which can be retrieved via node.approximate() or node.marginal().
/// Samples are accumulated column-wise, so that sampling n times from a distribution of dimension p
/// will generate a p x n matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomWalk {

    pub pos : usize,

    pub traj : DMatrix<f64>,

    pub weights : DVector<f64>

}

impl RandomWalk {

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

    /// Retrieves the current
    pub fn state<'a>(&'a self) -> DVectorSlice<'a, f64> {
        self.traj.column(self.pos)
    }

    pub fn histogram(&self, ix : usize) -> Option<Histogram> {
        if ix < self.traj.nrows() {
            let samples = self.traj.row(ix).clone_owned().transpose();
            Some(Histogram::build(&samples))
        } else {
            None
        }
    }

}

/// Builds a random walk from an external algorithm output, giving equal
/// weight to any sample. The output is assumed to be organized column-wise,
/// in the same representation hold by RandomWalk.
impl<S> From<Matrix<f64, Dynamic, Dynamic, S>> for RandomWalk
where
    S : ContiguousStorage<f64, Dynamic, Dynamic>
{

    fn from(m : Matrix<f64, Dynamic, Dynamic, S>) -> Self {
        let traj = m.clone_owned();
        let pos = traj.ncols();
        let weights = DVector::from_element(pos, 1. / pos as f64);
        Self { traj, pos, weights }
    }
}

/*/// RandomWalk is implemented by distributions who may
/// maintain a history of changes in the natural parameter
/// scale of its parent(s) node(s).
/// TODO move to posterior trait.
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
    fn marginal(&self) -> Option<Marginal>;

    /*/// Runtime description for the implementor.
    fn description() -> SourceDistribution;*/

}*/


