pub mod ffi;

use nalgebra::DVector;

/*
pub struct LocationExponential {
    loc : f64,
    obs : f64
}

pub struct ScaledExponential {
    loc : f64,
    scale : f64,
    obs : f64
} */

/*/// Represents an ordered sequence of distribution realizations, that are independent when they have no
/// parent factors, and when they do they are conditionally-independent given the parent factor.
/// The Joint<D> is used to represent convergent graphs without reference-counting.
/// While having many independent realizations sharing Rc<RefCell<ParentFactor>> is one way to do it,
/// we miss out on Rust static memory safety guarantees and the possibility to vectorize calculations.
/// Having impl Condition<F> for Joint<D> lets us represent a generic directed graph with as tree, since
/// a convergent many-to-one relationship is collapsed into a one-to-one relationship. Conceptually,
/// from the user point-of-view and in the exported API, we still have a generic DAG,
/// but in terms of the data structure, it is represented as a tree, with a single top-level likelihood
/// node and the branches diverging to form the scale and location factors. Rust static memory guarantees
/// hold only for tree-like data structures, not generic DAGs.
pub struct Join<D>
where
    D : Distribution
{

    obs : DVector<f64>,

    loc : DVector<f64>,

    scale : DVector<f64>,

    d : PhantomData<D>

}*/

// impl From<[D]> for Joint<D>
// impl FromIterator<D> for Joint<D>

// Implemented only for univariate distributions. May also be called Univariate.
// If Distribution does not have a scale parameter, scale(&self) always return 1.0.
pub trait Exponential {

    type Location;

    type Scale;

    fn location(&self) -> Self::Location;

    fn scale(&self) -> Option<Self::Scale>;

}

enum Factor {

    Fixed(DVector<f64>, )
}

pub mod prob;

pub mod fit;

pub mod approx;

pub mod calc;

use std::io;
use std::fmt;
use std::fs;
use either::Either;

#[test]
fn condition() {

    use crate::prob::Normal;
    use crate::fit::Likelihood;

    let a : Box<[Normal]> = [0.0].iter()
        .map(|v| Normal::likelihood([v]) )
        .collect();
}

// extern "C" fn avg()
