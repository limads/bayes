use nalgebra::*;
use crate::prob::*;
use std::ops::AddAssign;
use serde::{Serialize, Deserialize};
use std::ops::Index;
use std::ops::Range;
use super::*;

/*/// Marginal is a collection of Histograms resulting from marginalization of a complex multivariate distribution.
/// Each index (which can be recovered by position or name) represents the result of marginalizing a distribution
/// over all variables but the one of interest.
pub struct Marginal {

    /// Applies link function to eta_traj to get this field.
    _theta_traj : DMatrix<f64>

}

impl Marginal {

    pub fn new(_theta_traj : DMatrix<f64>) -> Self {
        Self{ _theta_traj }
    }

    pub fn len(&self) -> usize {
        unimplemented!()
    }

    pub fn at(&self, ix : usize) -> Histogram {
        unimplemented!()
    }

    pub fn marginals(&self, a : usize, b : usize) -> MarginalHistogram {
        MarginalHistogram::build(self.at(a), self.at(b))
    }

    pub fn joint(&self, a : usize, b : usize) -> SurfaceHistogram {
        //SurfaceHistogram::build(&self, self.at(b))
        unimplemented!()
    }

}

impl Index<usize> for Marginal {

    type Output = Histogram;

    fn index(&self, _ix: usize) -> &Self::Output {
        unimplemented!()
    }

}

impl Index<Range<usize>> for Marginal {

    type Output = Vec<Histogram>;

    fn index(&self, _ix : Range<usize>) -> &Self::Output {
        unimplemented!()
    }

}*/


