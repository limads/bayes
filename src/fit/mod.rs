use std::borrow::Borrow;
use std::iter::{IntoIterator, FromIterator};
use crate::prob::MultiNormal;
use crate::prob::Categorical;
use rand::seq::IteratorRandom;
use std::collections::HashMap;
use nalgebra::{DVector, DVectorSlice};
use std::fmt::{self, Display, Formatter};

pub mod linear;

// KMeans, KNearest, ExpectMax
pub mod cluster;

// VE, BP
pub mod graph;

// OLS, WLS, BLS, IRLS
// pub mod linear;

// Metropolis
// pub mod markov;

// LMS, RLS, Kalman
// pub mod stochastic;

// We "read" data into the distributions using the most generic possible iterator over borrowable
// values. This allows the user to pass iterators by value or by reference. This is done to avoid
// "double copies" situations. For example, were we requiring iterators over vectors, the user
// would need to do a first copy to the vector then a second copy to the distribution. Using the
// current generic implementation, the user can pass, for example, a map over rows of CSV files,
// or a map over rows returned by a database driver, in such that the first occurence that the
// data is contiguous in memory is already the data buffer of the distribution, and can be accessed
// by distribution.values().

// TODO perhaps add ancestral trait Fit(.) which is implemented by all "Likelihood" sub-traits.

// Although any pair that implements Conditional will form a valid probabilistic graph,
// implementors of the fit(.)
// A distribution that serves the role of an independent sample conditioned on a parameter value.
// p(y|\theta), conditional or not on a prior parameter value. This serves to instantiate observable
// distributions in a graph.
pub trait Likelihood {

    type Observation;

    fn likelihood<S, O>(sample : S) -> Self
    where
        S : IntoIterator<Item=O>,
        O : Borrow<Self::Observation>;

    // Just modifies the distribution, by consuming self.values() and setting
    // the location (and possibly scale) paraments to the MLE estimates.
    fn fit(&mut self) {
        unimplemented!()
    }

}

// The user can now implement his estimation method by building a direct or indirect
// subtrait of Likelihood.
pub struct ConjugateError;

// Perhaps rename to JointLikelihood.
pub trait ConjugateLikelihood<O>
where
    Self : Likelihood<Observation=O>
{

    type Posterior;

    fn fit(&mut self) -> Result<&Self::Posterior, ConjugateError> {
        unimplemented!()
    }

}

/// A distribution whose conditional expectation is function of a linear
/// combination of non-random (fixed) values. This is the same as assuming
/// the (random, fixed) joint is multivariate normal, and y was fixed at y|x.
/// Use the builder pattern to "fix" the likelihood y at the x value. FixedLikelihoods
/// store not only the random and fixed data, but also a ref-counted common coefficient
/// vector. Perhaps rename to ConditionalLikelihood. Instead of estimating p(y|\theta),
/// fixed likelihoods work instead with estimating the transformation p(y - g(w0+x_iw_i)|(w_0,w_i)), where
/// w is the object of inference and g is a fixed function for each family of p (depending exclusively on
/// the type of variable y is).
pub trait FixedLikelihood<const N : usize> {

    // Fix this distribution at the given informed values.
    fn fix<'a, S, O>(&'a mut self, values : &[f64; N]) -> &'a mut Self;

    // Returns regression coefficients of linearized expected value
    fn coefficients<'a>(&'a self) -> &'a [f64; N];

    // Returns values at which distribution was fixed
    fn fixed<'a>(&'a self) -> &'a [f64; N];

    // TODO return MultiNormal<N>, where if the user did not inform a prior,
    // a new distribution is allocated.
    fn fit(&mut self) -> Result<&MultiNormal, ConjugateError> {
        unimplemented!()
    }

    // Returs the intercept.
    // self.bias()

    // Returns the coefficients.
    // self.coefficients()

}

/// Marginal likelihoods hold not only its random data, but a common ref-counted
/// allocation probability vector and a specific allocation index, corresponding to the
/// distribution index in the user-supplied vector.
/// Example:
/// let mut ms = [Normal::likelihood([y1, y2, y3]), Normal::likelihood([y1, y2, y3])]
/// ms.condition(Categorical::likelihood([0, 1]).map(|ix| Bounded::try_from(ix).unwrap() ))
/// This trait can be considered the discrete counterpart to StochasticLikelihood.
pub trait MarginalLikelihood<const N : usize> {

    fn marginal_probability<'a, S, O>(&'a mut self, prob : f64) -> &'a mut Self;

    fn view_marginal_probability(&self) -> f64;

    // TODO return Categorical<N>, where if the user did not inform a prior,
    // a new distribution is allocated.
    fn fit(&mut self) -> Result<&Categorical<N>, ConjugateError> {
        unimplemented!()
    }

}

/*/// A distribution that represents the (probabilistic) weights of a linear combination of constant values.
/// self.fixed caches the informed values; although the distribution really is a 1-D distribution over
/// a model weight vector. It can be shown those coefficients are distributed as a multinormal in the limit,
/// which is why MultiNormal implements Fixed. Normal also represents fixed for the case of a single fixed/coefficient pair.
/// Regression problems with N dependent variables are
/// represented with coefficients stored at [MultiNormal; N], where N is the number of dependent variables, which also implement Fixed.
/// Fixed multinormals may have a random location parameter; But their covariance is always fixed, therefore
/// instantiating a fixed multinormal means that only the left-hand side of the graph might be
pub trait Fixed {

    type Observation;

    type Posterior;

    fn fixed<S, F, O>(sample : S, fixed : &[impl IntoIterator<Item=F>]) -> Self
    where
        S : IntoIterator<Item=O>,
        O : Borrow<Self::Observation>,
        F : Borrow<f64>;

    fn set_fixed(&mut self, fixed : &[impl Borrow<f64>]);

    fn fit(&mut self) -> Result<&Self::Posterior, String> {
        unimplemented!()
    }

}*/

/*/// A distribution that serves the role of a marginalized likelihood function,
/// p(y) = \sum_i p(y|z_i)p(z_i), necessarily conditional on a binary or categorical discrete factor z_i.
pub trait Marginal {

    type Observation;

    type Posterior;

    fn marginal<S, O>(sample : S, clusters : usize) -> Self
    where
        S : IntoIterator<Item=O>,
        O : Borrow<Self::Observation>;

    fn set_marginal(&mut self, cluster : usize);

    fn fit(&mut self) -> Result<&Self::Posterior, String> {
        unimplemented!()
    }
}*/

/// A distribution in which its prior can be interpreted as a continuous latent state. Might also be called StochasticLikelihood
/// or StateLikelihood.
/// Markov dependencies can either be directed (chains) or undirected (fields)
/// The method self.evolve automatically forgets any values, simply updating its latent state instead.
/// This trait can be considered the continous counterpart to MarginalLikelihood. This is
/// useful to perform inference on Markov-like processes; Where knowledge of a transition
/// model and a prior makes posterior inference on a hidden state possible by conditioning
/// the current variate on the previous process realization.
pub trait MarkovLikelihood {

    type Observation;

    type State;

    type Posterior;

    fn stochastic<S, O>(order : usize) -> Self
    where
        S : IntoIterator<Item=O>,
        O : Borrow<Self::Observation>;


    // Just call this fit(.) as well.
    // fn evolve(&mut self, obs : &Self::Observation);

    fn state(&self) -> &Self::State;

}

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


