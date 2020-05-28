use nalgebra::*;
use crate::distr::*;
use super::*;
use crate::optim::*;
use std::ops::AddAssign;

/// Expectation Maximization algorithm, for approximating conditional posteriors. At each
/// phase of the algorithm, we pick a node to optimize; while maintaining all other nodes constant.
/// By iterating over non-root nodes of the graph iteractively; and optimizing the node gradient with
/// respect to the full graph log-probability, we are guaranteed to arrive at the posterior mode
/// if the negative log-posterior is convex. The algorithm outputs a (local) normal approximation by using the last gradient
/// steps to estimate the precision matrix. The global mode of the posterior is given by this gaussian
/// mode; and this gaussian precision is a good approximation to the mode precision if we do not
/// go too far in the parameter space (in the natural parameter scale). The returned graph can also
/// be interpreted as the conditional posterior for the last optimized node conditional on all
/// other nodes held constant at their modes. The posterior mode can be used to make predictions
/// conforming to some decision rule; or to build a proposal distribution for the Metropolis algorithm.
/// This proposal will be a multivariate gaussian with mean defined as the concatenated node means; and
/// covariance defined only at the diagonal for single-parameter nodes or at the block-diagonal for
/// multiparameter nodes. The node-specific gaussian approximation can be obtained by node.approximate()
/// at the graph returned by ExpectMax::fit.
pub struct ExpectMax {

}

impl<D> Estimator<D> for ExpectMax
    where D : Distribution
{

    fn fit<'a>(&'a mut self, y : DMatrix<f64>) -> Result<&'a D, &'static str> {
        unimplemented!()
    }

}

