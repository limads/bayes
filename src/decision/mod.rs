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

/// A decision boundary is the output of an optimized decision process
/// to select between two alternative probabilistic models, after
/// considering an error criterion.
/// The difference in log-probability between a model
/// and an alternative lies on the real line (where zero means indifference to which model
/// is best considering that false positives are as bad as false negativas).
/// The decision boundary is a partition of this line away from zero in either
/// direction that gives more weight to false positives relative to false negatives.
///
/// While useful as a final output of an inference procedure, DecisionBoundary(.) also
/// implements distribution, and so can be composed with other distributions inside a graph
/// (behaving as a Bernoulli random varialble that decides if
/// the left hand branch is best than the right hand branch given the informed error criterion).
///
/// Decision is a Distribution and Likelihood implementor that behaves differently:
/// All the data interfacing with it is passed through it to the next Likelihood implementor,
/// and the internal (scalar) parameter value is determined by the log-likelihood evaluation
/// of the factor. The log-likelihood of the factor is taken to be the logit of this element. This
/// allows the user to use a familiar interface of sampling and parameter calculation for the decision
/// process.
pub struct Decision<'a> {
    // In the sample(.) forward pass through the graph, the samples from the left branch are transformed
    // via a user-defined function to the fixed Bernoulli parameters, and those parameters are used
    // to evaluate if the incoming transformed samples from the right branch satisfy the boundary
    // established by the criterion;
    // In the log_prob(.) backward pass, the incoming
    // sample has its log-probability calculated relative to the fixed decision vector (which is passed to the right)
    // and the fixed decision vector log_probability is passed to the left.

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

impl<'a> Decision<'a> {

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

    fn _auc() -> f64 {
        unimplemented!()
    }

    /// Returns the actual estimated error rate from the informed sample. This quantity is
    /// supposed to be as close to
    fn _error_rate(&'a self) -> ErrorRate {
        unimplemented!()
    }

}

/*impl Distribution for Decision {

}

impl Likelihood for Decision {

    // This implementor just takes a series of binary outcomes
    // and propagate them to the internal Bernoulli variable.
    // If there are data-dependent factors that also implement Likelihood,
    // The data is passed to this implementor without any loss.

}*/

/// BayesFactor can be used to compare posteriors to arbitrary analytical
/// distributions (Null or saturated); or to compare the same posterior
/// with itself at different values by comparing their conditional log-posteriors.
/// It is built from a pair of distributions of the same kind, and outputs a Decision.
/// The decision that is output as the comparison can be used as a component of a probabilistic graph,
/// since decision behaves as a Bernoulli variable that uses the log-likelihood as its natural parameter
/// (logit) value.
///
/// ```rust
/// // let bf = m1.compare(m2);
///
/// // Verify if self is more likely than the alternative relative to the informed sample
/// // and the default error criterion (indiference to false positives vs. false negatives);
/// // bf.best(y, default());
///
/// // Obtain optimized decision boundary over the log-posterior for the given criterion.
/// // let bound = bf.optimize(y, crit, true_values);
/// ```
pub struct BayesFactor<'a, D, E>
    where
        D : Distribution,
        E : Distribution
{

    a : &'a D,

    b : &'a E,

    bound : Option<Decision<'a>>
}

impl<'a, D,E> BayesFactor<'a, D, E>
    where
        D : Distribution,
        E : Distribution
{

    pub fn log_diff(&self, y : DMatrixSlice<'_, f64>, x : Option<DMatrixSlice<'_, f64>>) -> f64 {
        self.a.log_prob(y.clone(), x.clone()) - self.b.log_prob(y, x)
    }

    /// Decision boundary accepts Default::default() for a standard
    /// cost to positive/negative errors.
    pub fn best(
        &'a self,
        _y : &'a DMatrix<f64>,
        _boundary : Decision<'a>
    ) -> bool {
        unimplemented!()
    }

    /// Calls self.best(.) iteratevely, changing a scalar that partitions the model log-posterior differences
    /// until the decisions (taken not at zero, but at the optimized value) match the observed
    /// 0/1 decision vector as close as possible given the desired criterion (potentially after applying a transformation f to the sample).
    /// f(.) is any function that maps the potentially continuous outcomes to the 0/1
    /// domain (this is just identity if the Bernoulli).
    /// Models for which the output
    /// is a categorical can define a decision rule as some linear combination of the categories
    /// (for example, an ordered outcome is a categorical output summed up to the kth element compared
    /// against all other elements). Models with univariate or multivariate continuous outcomes can
    /// determine arbitrary step functions of those to yield an output. Future decisions are then not made at
    /// zero, but at the chosen decision boundary.
    /// The method also calculates the empirical Error Rates, which
    /// should be as close to the ideal criterion informed by the user as possible. The lifetime of the
    /// boundary becomes tied to the lifetime of the sample used to calculate it.
    pub fn optimize(
        &'a self,
        _y : DMatrix<f64>,
        _criterion : ErrorRate,
        _outcomes : DVector<f64>,
        _f : &'a dyn Fn(DMatrix<f64>)->DVector<f64>
    ) -> Decision<'a> {
        unimplemented!()
    }

    pub fn new(a : &'a D, b : &'a E) -> Self {
        Self{ a, b, bound : None }
    }

}


