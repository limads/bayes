use std::borrow::Borrow;

pub mod sample;

/* #[cfg(feature="sp")]
type Variate = f32;

#[cfg(feature="dp")]
type Variate = f64;

// Trait shared by analytical (Exponentials, Joints) and approximated (Histogram, Walk) distributions.
pub trait Distribution {

    fn mean(&self) -> f64;

    fn mode(&self) -> f64;

    fn var(&self) -> f64;

    fn cov(&self, other : &Self) -> Option<f64>;

}

// A mode vector or scalar. Generic distributions default to crate::Variate.
pub struct Mode(DVector<f64>);

// A mean vector or scalar
pub struct Mean(DVector<f64>);

// A full posterior distribution representation, in analytical form or approximated (with approx::Histogram or approx::Walk).
pub struct Marginal<T>(

pub trait Estimator {

    // Target is either mode, mean or marginal.
    type Target;

    // What should the estimator output when it fails.
    type Error;

}*/

pub mod linear;

// KMeans, KNearest, ExpectMax
pub mod cluster;

// VE, BP
pub mod graph;

/// General-purpose trait implemented by actual estimation algorithms.
pub trait Estimator
where Self : Sized {

    type Settings;

    type Error;

    fn estimate(
        sample : impl Iterator<Item=impl Borrow<[f64]>> + Clone,
        settings : Self::Settings
    ) -> Result<Self, Self::Error>;

}

/* Gibbs-style conditional posterior estimator.
Implement generic gibbs estimation: estimate from A, then modify
B given the current state of A, then estimate B (then iterate). Can
be nested indefinitely.
pub struct Gibbs<A, B>
where
    A : Estimator,
    B : Estimator
{
    a : A,
    b : B,
    b_cond_a : Box<dyn Fn(&A, &mut B)>,
    a_cond_b : Box<dyn Fn(&mut A, &B)>
}

// Gelman-style multilevel models. Estimate from all AS, then use
// their results as the likelihood for B, then estimate B. Can be nested
// indefinitely, up to a root model, after which the whole process is repeated.
pub struct Gelman<A, B>
where
    A : Estimator,
    B : Estimator
{
    b : B,
    as : Vec<A>,
    b_cond_as : Box<dyn Fn(as : &[A], b : &mut B)>
}
*/

