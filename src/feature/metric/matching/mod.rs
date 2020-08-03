use super::*;
use crate::optim::{Optimizer, Minimum};
use std::collections::HashMap;

/// Unlike clustering algorithms, match algorithms must interpret the
/// Sample vector in a specific way to optimize a subset of the vector.

/// A match represents the result of the matching process
/// between a pair of features. This match carries both
/// a metric and extra parameters that quality the match,
/// such as a global relative translation or scale.
pub struct Match<M>
    where
        M : Metric
{
    m : M,

    /// Each minima is a functional (identifiable) best-match criterion that
    /// can be optimized over. Must identify a set of columns of the sample
    /// with respect to which the match was calculated.
    min : HashMap<usize, Minimum>,
}

/*pub trait Matching<M, O>
where
    M :  Metric,
    O : Optimizer
{

    fn best_match(a : S, b : T) -> Match
    where
        S : Sample,
        T : Sample;


}*/

