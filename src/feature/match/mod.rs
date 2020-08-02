use super::metric::*;
use crate::optim::gsl::{Optimizer, Minima};
use std::collections::HashMap;

pub struct Match<T>
where
    T : Eq + Hash
{

    /// Each minima is a functional (identifiable) best-match criterion that
    /// can be optimized over.
    min : HashMap<T, Minima>,

}

pub trait Matching<M, O>
where
    M :  Metric,
    O : Optimizer
{

    fn best_match(a : S, b : T) -> Match
    where
        S : Sample,
        T : Sample;


}


    
