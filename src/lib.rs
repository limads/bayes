use nalgebra::DVector;

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
