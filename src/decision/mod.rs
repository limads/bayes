use nalgebra::*;
use crate::distr::*;
use std::default::Default;

/// Error rate, used by the user to obtain an optimized decision
/// boundary; or to store empirical decision boundaries after
/// a decision process has been taken. true_pos + false_neg should
/// sum to one; and true_neg + false_pos should also sum to one.
pub struct ErrorRate {

    pub true_pos : f64,

    pub true_neg : f64,

    pub false_pos : f64,

    pub false_neg : f64
}

impl Default for ErrorRate {

    fn default() -> Self {
        Self {
            true_pos : 0.5,
            true_neg : 0.5,
            false_pos : 0.5,
            false_neg : 0.5
        }
    }

}

/// A decision boundary is the output of a model comparison strategy
/// and some sample (which can be, but is not required to, be the sample
/// used to fit the model). It is an optimized strategy to make a decision
/// with respect to a pair of probabilistic models and a distribution.
/// If the difference between a model log_probability and an alternative model
/// log_probability live on a line (where zero means indifference to which model
/// is best considering that false positives are as bad as false negativas),
/// the decision boundary is a partition of this line away from zero in either
/// direction that gives more weight to false positives relative to false negatives.
/// While useful as a final output of an inference procedure, DecisionBoundary(.) also
/// implements distribution, and so can be composed with other distributions inside a graph
/// (behaving as a Bernoulli random varialble that decides if
/// the left hand branch is best than the right hand branch given the informed error criterion).
/// Given a realized decision process with fixed ideal error rate criterion
/// and fixed known decision vector,
/// the decision boundary behaves as a Bernoulli random variable with probability
/// determined only by outputs from one of the branches, that is assumed a fixed decision vector
/// relative to the other.
/// In the sample(.) forward pass through the graph, the samples from the left branch are transformed
/// via a user-defined function to the fixed Bernoulli parameters, and those parameters are used
/// to evaluate if the incoming transformed samples from the right branch satisfy the boundary
/// established by the criterion;
/// In the log_prob(.) backward pass, the incoming
/// sample has its log-probability calculated relative to the fixed decision vector (which is passed to the right)
/// and the fixed decision vector log_probability is passed to the left.
pub struct DecisionBoundary<'a> {

    /// Sample for which this decision
    _sample : &'a DMatrix<f64>,

    /// Single point over the log-likelihood difference between two models
    _log_lik : f64,

    _ideal_rate : ErrorRate,

    /// Empirical error rate, after the boundary has been optimized over a sample.
    /// empirical_rate should be as close as possible to ideal_rate given the sample
    /// and the pair of models used to make decisions over the sample.
    _empirical_rate : ErrorRate,

}

impl<'a> DecisionBoundary<'a> {

    /// Creates a new decision boundary over the informed sample,
    /// by trying to approach the ideal Error Rate as close as possible. If all missing criteria
    /// are equally important, use ErrorRate::default() (which yields (0.5, 0.5, 0.5, 0.5)).
    pub fn new(_y : &'a DMatrix<f64>, _ideal : ErrorRate) -> Self {
        unimplemented!()
    }

    fn _d_prime() -> f64 {
        unimplemented!()
    }

    fn _roc() -> f64 {
        unimplemented!()
    }

    /// Returns the actual estimated error rate from the informed sample. This quantity is
    /// supposed to be as close to
    fn _error_rate(&'a self) -> ErrorRate {
        unimplemented!()
    }

}

/// BayesFactor can be used to compare posteriors to arbitrary analytical
/// distributions (Null or saturated); or to compare the same posterior
/// with itself at different values by comparing their conditional log-posteriors.
/// A peak detection problem, for example, can be formulated
/// as:
/// m1.iter_factors().next().try_set(&[0.]);
/// m2.iter_factors().next().try_set(&[1.]);
/// Where m1 and m2 are identical posteriors fitted in the same dataset,
/// and the unique factor is a Bernoulli.
/// Then:
/// let bf = m1.compare(m2)
/// Allows some model comparison strategies:
/// bf.best(sample, default()) tells if self is more likely than other relative to the informed unobserved sample
/// and the default error criterion (indiference to false positives vs. false negatives);
/// bf.optimize(sample, crit, |x| { lookup[x] }, true_values) returns a decision boundary over the log-posterior
/// diference between self and other that satisfy the required error rate criterion. The informed function maps the
/// sample values to a decision 0/1 space so it can be compared to true_values.
pub struct BayesFactor<'a, D, E>
    where
        D : Distribution,
        E : Distribution
{

    _a : &'a D,

    _b : &'a E,

    _bound : DecisionBoundary<'a>
}

impl<'a, D,E> BayesFactor<'a, D, E>
    where
        D : Distribution,
        E : Distribution
{

    /// Decision boundary accepts Default::default() for a standard
    /// cost to positive/negative errors.
    pub fn best(
        &'a self,
        _y : &'a DMatrix<f64>,
        _boundary : DecisionBoundary<'a>
    ) -> bool {
        unimplemented!()
    }

    /// This method calls self.best(.) iteratevely, changing a scalar that partitions the model log-posterior differences
    /// until the decisions (taken not at zero, but at the optimized value) match the observed
    /// 0/1 decision vector as close as possible given the desired criterion (potentially after applying a transformation f to the sample).
    /// f(.) is any function that maps the potentially continuous outcomes to the 0/1
    /// domain (this is just identity if the Bernoulli). Models where the output
    /// is a categorical can define a decision rule as some linear combination of the categories
    /// (for example, an ordered outcome is a categorical output summed up to the kth element compared
    /// against all other elements). Models with univariate or multivariate continuous outcomes can
    /// determine arbitrary step functions of those to yield an output. Future decisions are then not made at
    /// zero, but at the chosen decision boundary. The method also calculates the empirical Error Rates, which
    /// should be as close to the ideal criterion informed by the user as possible. The lifetime of the
    /// boundary becomes tied to the lifetime of the sample used to calculate it.
    pub fn optimize(
        &'a self,
        _y : DMatrix<f64>,
        _criterion : ErrorRate,
        _outcomes : DVector<f64>,
        _f : &'a dyn Fn(DMatrix<f64>)->DVector<f64>
    ) -> DecisionBoundary<'a> {
        unimplemented!()
    }

    pub fn new(_a : &'a D, _b : &'a E) -> Self {
        unimplemented!()
    }
}


