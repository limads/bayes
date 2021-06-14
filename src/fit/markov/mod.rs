use nalgebra::*;
use nalgebra::storage::Storage;
use crate::prob::*;
use std::ops::AddAssign;
use serde::{Serialize, Deserialize};
use std::ops::Index;
use nalgebra::storage::*;
use crate::approx::Histogram;

// Metropolis-Hastings posterior sampler (Work in progress).
mod metropolis;

pub use metropolis::*;

/// A sequence of natural parameter iterations. The distribution at the current node
/// holds the parameter trajectory of all distributions at the parent nodes, which during
/// optimization or posterior sampling are considered as conditioned or unconditional priors.
/// After optimization/simulation, this trajectory is used to build an approximation to the
/// corresponding posterior entry, which can be retrieved via node.approximate() or node.marginal().
/// Samples are accumulated column-wise, so that sampling n times from a distribution of dimension p
/// will generate a p x n matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {

    pub pos : usize,

    pub traj : DMatrix<f64>,

    pub closed : bool

}

/// Builds a random walk from an external algorithm output, giving equal
/// weight to any sample. The output is assumed to be organized column-wise,
/// in the same representation held by RandomWalk.
impl<S> From<Matrix<f64, Dynamic, Dynamic, S>> for Trajectory 
where
    S : Storage<f64, Dynamic, Dynamic>
{
    fn from(mat : Matrix<f64, Dynamic, Dynamic, S>) -> Self {
        Trajectory{ traj : mat.clone_owned(), pos : mat.ncols(), closed : true }
    }
}

impl Trajectory {

    pub fn new(len : usize, dim : usize) -> Self {
        Self { traj : DMatrix::zeros(len, dim), pos : 0, closed : false }
    }
    
    pub fn copy_from(&mut self, data : DMatrixSlice<'_, f64>) {
        self.traj.copy_from(&data);
        self.pos = data.ncols();
        self.closed = true;
    }
    
    /// Expands this trajectory, assuming the given element was expanded n times.
    pub fn expanded_samples(&self, weights : &[usize]) -> DMatrix<f64> {
        let final_sz = weights.iter().sum();
        let mut out = DMatrix::zeros(final_sz, self.traj.nrows());
        let mut curr_ix = 0;
        let mut curr_count = 0;
        for mut row_out in out.row_iter_mut(){
            row_out.copy_from(&self.traj.row(curr_ix));
            curr_count += 1;
            if weights[curr_ix] == curr_count {
                curr_ix += 1;
            } 
        }
        out
    }
    
    pub fn close(mut self) -> Self {
        self.closed = true;
        let curr_pos = self.pos;
        let traj_sz = self.traj.nrows();
        if self.traj.nrows() > self.pos+1 {
            self.traj = self.traj.remove_rows(curr_pos, traj_sz);
        }
        self
    }
    
    /*pub fn new(start : DVectorSlice<'_, f64>) -> Self {
        let mut traj = DMatrix::zeros(start.nrows(), 1000);
        let weights = DVector::from_element(1000, 1.);
        let pos = 0;
        traj.column_mut(0).copy_from(&start);
        Self{ pos, traj, weights }
    }*/

    /*pub fn step(&mut self, opt_data : Option<DVectorSlice<'_, f64>>) {
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
    }*/

    /*pub fn step_increment(&mut self, incr : DVectorSlice<'_, f64>) {
        let prev : DVector<f64> = self.traj.column(self.pos).into();
        self.traj.column_mut(self.pos+1).copy_from(&prev);
        self.traj.column_mut(self.pos+1).add_assign(&incr);
        self.pos += 1;
    }*/

    /// Retrieves the current step value.
    pub fn state<'a>(&'a self) -> DMatrixSlice<'a, f64> {
        let dim = self.traj.ncols();
        self.traj.slice((self.pos, 0), (1, dim))
    }
    
    /// Retrieves the previous step value
    pub fn prev_state<'a>(&'a self) -> DMatrixSlice<'a, f64> {
        assert!(self.pos >= 1);
        let dim = self.traj.ncols();
        self.traj.slice((self.pos - 1, 0), (1, dim))
    }
    
    /// Retrieves a mutable reference to the current step
    pub fn state_mut<'a>(&'a mut self) -> DMatrixSliceMut<'a, f64> {
        let dim = self.traj.ncols();
        let pos = self.pos;
        self.traj.slice_mut((pos, 0), (1, dim))
    }
    
    /// Increments the current step and retrives a mutable reference to the new current state.
    /// This method can be used in conjunction with distr1.sample_mut(distr2.trajectory_mut().state_mut());
    /// Sampling column-wise (into a wide matrix) makes it easier to calculate log-probabilities later,
    /// because we can pass the column slice as a natural parameter. But sampling row-wise makes it easier
    /// to use the sample_into(.) API, which assumes samples always comes as sequential rows. Perhaps
    /// we can adopt the convention that whenever n=1, we accept sampling to conform to whichever
    /// structure we receive, be it a matrix row or matrix column. Also, if this is a wide matrix,
    /// we can easily do slice::copy_within to move the trajectory around.
    pub fn step<'a>(&'a mut self) -> DMatrixSliceMut<'a, f64> {
        if !self.closed {
            self.pos += 1;
            if self.pos == self.traj.nrows() - 1 {
                self.closed = true;
            }
            let pos = self.pos;
            let ncols = self.traj.ncols();
            self.traj.slice_mut((pos, 0), (1, ncols))
        } else {
            panic!("Tried to update closed trajectory");
        }
    }

    pub fn trim_begin(mut self, n : usize) -> Self {
        assert!(self.closed);
        self.traj = self.traj.remove_rows(0, n);
        self.pos -= n;
        self
    }
    
    /// Builds a Histogram over a single parameter value in this trajectory.
    pub fn histogram(&self, ix : usize, n_bins : usize) -> Option<Histogram> {
        /*if ix < self.traj.nrows() {
            let samples = self.traj.row(ix)
                .columns(0, self.pos)
                .iter()
            Some(Histogram::calculate(samples, n_bins))
        } else {
            None
        }*/
        unimplemented!()
    }

}

/*impl<S> From<Matrix<f64, Dynamic, Dynamic, S>> for Trajectory
where
    S : ContiguousStorage<f64, Dynamic, Dynamic>
{

    fn from(m : Matrix<f64, Dynamic, Dynamic, S>) -> Self {
        let traj = m.clone_owned();
        let pos = traj.ncols();
        let weights = DVector::from_element(pos, 1. / pos as f64);
        Self { traj, pos, weights }
    }
}*/

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

/* Sample from d and apply the transformation f to the output results, in order to estimate
the quantity f(d) via the Monte-Carlo method. out is assumed to be of dimension m*n where
m is the dimensionality of the distribution and n is the number of samples to be taken.
fn simulate(d : &dyn Distribution, f : Fn(&[f64], &mut [f64]), out : &mut DMatrix<f64>) {
    let distr_dim = d.view_parameter().nrows();
    let n_draws = out.len() / distr_dim;
    assert!(out.nrows() % n_draws == 0);
    for i in 0..n_draws {
        let out_slice = out.slice_mut((i*distr_dim, 0), (distr_dim, distr_dim));
        d.sample_into(out_slice);
        for row in out_slice {
            f(row, row);
        }
    }
}*/


