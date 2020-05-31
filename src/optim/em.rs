use nalgebra::*;
use crate::distr::*;
// use super::*;
// use crate::optim::*;
// use std::ops::AddAssign;

/// (Work in progress) The expectation maximization algorithm is a general-purpose inference
/// algorithm that generates a posterior gaussian approximation for each
/// distribution composing a model. At each step of the algorihtm, the full
/// model log-probability is evaluated by changing the parameter vector of a single node
/// in a probabilistic graph, while keeping all the others at their last estimated local
/// maxima. The algorithm is guaranteed to converge to a global maxima when the log-posterior
/// is a quadratic function. The gaussian approximation can be used directly if only the
/// posterior mode is required (i.e. prediction; decision under a quadratic loss) or can
/// serve as a basis to build an efficient proposal distribution for the Metropolis-Hastings posterior
/// sampler, which can be used to build a full posterior.
pub struct ExpectMax {

}

impl<D> Estimator<D> for ExpectMax
    where D : Distribution
{

    fn fit<'a>(&'a mut self, _y : DMatrix<f64>) -> Result<&'a D, &'static str> {
        unimplemented!()
    }

}

