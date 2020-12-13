use nalgebra::*;
use nalgebra::storage::*;
use std::fmt::Debug;
use std::ops::AddAssign;
use crate::model::decision::BayesFactor;
use std::fmt::Display;
use crate::fit::walk::Trajectory;
use anyhow;
use thiserror::Error;
use crate::sample::*;
use std::slice;
use either::Either;
use std::iter;
use std::collections::HashMap;

/// Structure to represent one-dimensional empirical distributions non-parametrically (Work in progress).
mod histogram;

pub use histogram::*;

/// Collection of histograms, product of marginalization.
mod marginal;

pub use marginal::*;

mod poisson;

pub use poisson::*;

mod beta;

pub use beta::*;

mod bernoulli;

pub use bernoulli::*;

mod gamma;

pub use gamma::*;

mod normal;

pub use normal::*;

mod multinormal;

pub use multinormal::*;

mod wishart;

pub use wishart::*;

mod categorical;

pub use categorical::*;

mod dirichlet;

pub use dirichlet::*;

mod vonmises;

pub use vonmises::*;

/// Trait shared by all parametric distributions in the exponential
/// family. The Distribution immediate state is defined by a parameter vector
/// (stored  both on a natural and canonical parameter scale) which can
/// be changed and inspected via set_parameter/view_parameter. Distribution
/// summaries (mean, mode, variance and covariance) can be retrieved based
/// on the current state of the parameter vector. The distribution sampling/
/// log-probability methods are dependent not only on the current parameter vector,
/// but also on the state of any applied conditioning factors.
///
/// Distributions of the bayes crate may carry scalar or vector-valued parameter values,
/// which facilitate their use as conditional expectations and/or to represent multivariate
/// sampling and calculation of log-probabilities.
pub trait Distribution
    where Self : Debug + Display
{

    /// Returns the expected value of the distribution, which is a function of
    /// the current parameter vector.
    fn mean(&self) -> &DVector<f64>;

    // TODO transform API to:
    // "natural" is always the linear (or scaled in the normal case) parameter;
    // "canonical" is always the non-linear (or unscaled in the normal case) parameter.

    // view_natural() / set_natural()
    // view_canonical() / set_canonical()

    /// Acquires reference to internal parameter vector; either in
    /// natural form (eta) or canonical form (theta).
    fn view_parameter(&self, natural : bool) -> &DVector<f64>;

    /// Set internal parameter vector at the informed value; either passing
    /// the natural form (eta) or the canonical form (theta).
    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, natural : bool);

    /// Returns the global maximum of the log-likelihood, which is a function
    /// of the current parameter vector. Note that
    /// for bounded distributions (Gamma, Beta, Dirichlet), this value might just the the
    /// the parameter domain inferior or superior limit.
    fn mode(&self) -> DVector<f64>;

    /// Returns the dispersion of the distribution. This is a simple function of the
    /// parameter vector for discrete distributions; but is a constant or conditioning
    /// factor for continuous distributions. Returns the diagonal of the covariance
    /// matrix for continuous distributions.
    fn var(&self) -> DVector<f64>;

    /// Returns None for univariate distributions; Returns the positive-definite
    /// covariance matrix (inverse of precision matrix) for multivariate distributions
    /// with independent scale parameters.
    fn cov(&self) -> Option<DMatrix<f64>>;

    /// Returns the inverse-covariance (precision) for multivariate
    /// distributinos; Returns None for univariate distributions.
    fn cov_inv(&self) -> Option<DMatrix<f64>>;

    fn corr(&self) -> Option<DMatrix<f64>> {
        // D^{-1/2} self.cov { D^{-1/2} }
        unimplemented!()
    }

    // fn observations(&self) -> Option<&DMatrix<f64>> {
    //    unimplemented!()
    // }

    // Effectively updates the name of this distribution.
    // fn rename(&mut self);
    
    /// Evaluates the log-probability of the sample y with respect to the current
    /// parameter state, and optionally by transforming the random variable by the matrix x.
    /// This method just dispatches to a sufficient statistic log-probability
    /// or to a conditional log-probability evaluation depending on the conditioning factors
    /// of the implementor. The samples at matrix y are assumed to be independent over rows (or at least
    /// conditionally-independent given the current factor graph). Univariate factors require that
    /// y has a single column; Multivariate factors require y has a number of columns equal to the
    /// distribution parameter vector.
    fn log_prob(&self, y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>) -> f64;

    /// Sample from the current distribution; If the sampling unit has multiple dimensions,
    /// they are represented over columns; If multiple units are sampled (if there are multiple
    /// entries for the parameter vector), a variable number of rows is emitted. Samples should
    /// follow the same structure as the argument to log_prob(.).
    fn sample(&self) -> DMatrix<f64> {
        let n = self.mean().nrows();
        let mut m : DMatrix<f64> = DMatrix::zeros(n, 1);
        self.sample_into((&mut m).into());
        m
    }

    fn sample_into(&self, dst : DMatrixSliceMut<'_,f64>);

}

// Maybe move to bayes::Model, since this is related to model building?
/// A probabilistic graph is linked by each element holding an owned Distribution
/// to its immediate parent(s), which allows for fast and flexible
/// ways to walk over the graph. But some compile-time constraints on which distributions
/// can link to others is important to guarantee samples will
/// have valid column dimensionality and bounds. This trait guarantee minimal
/// compile time constraints on that, since condition(.) can be implemented only
/// for factors that yield samples of same dimensionality (univariate or multivariate)
/// and also can restrict potential exponential family factors to the respective
/// conjugate priors.
///
/// This trait is used to build probability distributions that factor in a directed and tree-like structure,
/// with a root likelihood node and branch/leaf nodes that are priors and hyperpriors (before inference)
/// and posterior factors after inference.
///
/// This trait is also the basis for the bayes JSON model definition: To parse the next step in the graph,
/// you have to verify that the JSON field D satisfies Conditional<D> for the current JSON node Self.
///
/// A parsed probabilistic model is always a binary tree with a root "predictive" or "likelihood" node,
/// with bound neighboring nodes satisfying this Conditional trait. This binary tree link location parameters to location
/// prior/posterior distributions, and possibly scale parameters to scale priors/posterior distributions.
///
/// The user can, however, specify probabilistic graphs
/// or arbitrary complexity, by building  joint distribution over continuous or discrete variables. Those graphs,
/// however, always resolve to one binary tree of conditional distributions (minimally having a Likelihood, but more
/// commonly having a Likelihood and a prior node), by joining joint nodes into a multivariate distribution. 
/// This binary tree grows when the user specify conditional expectation dependencies (by substituting a
/// constant in the mean field for another distribution).
pub trait Conditional<D>
    where
        Self : Distribution + Sized,
        D : Distribution
{

    /// Takes self by value and return a conditional distribution
    /// of self, with the informed factor as its parent.
    fn condition(self, d : D) -> Self;

    /// Returns a view to the factor.
    fn view_factor(&self) -> Option<&D>;

    /// Takes self by value and outputs its factor.
    fn take_factor(self) -> Option<D>;

    /// Returns a mutable reference to the parent factor. This is useful if you have to
    /// iterate over the factor graph in a way that preserves some state: Just call your
    /// function or closure recursively using factor_mut(.) as the argument at every call
    /// until the full graph is visited.
    fn factor_mut(&mut self) -> Option<&mut D>;
    
    // Searches the given factor by variable name, returning it if successful.
    // fn search_factor(&self, name : &str) -> Option<&D>;

}

// Maybe move to bayes::model, since this is related to model building.
/// Implemented by distributions which compose together to yield multivariate
/// joint distributions. Implementors are Normal(Normal)->MultiNormal and
/// MultiNormal(Normal)->MultiNormal for continuous variables; and Bernoulli->Bernoulli or
/// Categorical->Categorical for joint discrete distributions linked via conditional probability
/// tables (CPTs). This distribution is the basis to parse joint probabilistic models from JSON:
/// if two variable names are field names of a same JSON object, those names are assumed part of a 
/// joint distribution; the correlation or CPT argument define how those two elements will be linked.
pub trait Joint<D>
where
    Self : Distribution + Sized,
    // Self : Distribution + Sized,
    D : Distribution,
    Self::Output : Distribution
{

    type Output;

    /// Changes self by assuming joint normality with another
    /// independent distribution (extends self to have a block-diagonal
    /// covariance composed of the covariance of self (top-left block)
    /// with the covariance of other (bottom-right block). The parameter
    /// corr is used to specify the partial correlations between the parameters
    /// of the implementor and the parameter of the added element.
    /// (this would be the entry at the "standardized" precision matrix).
    fn joint(self, other : D, corr : Option<&[f64]>) -> Option<Self>;
}

// TODO maybe rename to 
/*
pub enum Condition<D> {
    Deterministic(DMatrix<f64>),
    Stochastic(D)
}
*/
/// Univariate factors can either have a conjugate distribution
/// factor (as all distribution implementors have) or a conditional
/// expectation factor: The sampling and log-prob of the distribution
/// holding this factor is calculated relative to a random draw from
/// the parent MultiNormal distribution, which is interpreted as a
/// set of natural parameters for each realization of this random variable.
/// Conjugate factors express stochastic relationships between a prior (
/// which might be an observed group label, unobserved prior assumtion
/// or marginal ML prior estimate) and the factor, i.e. sampling the factor
/// is conditional on sampling the conjugate parameter; CondExpect factors
/// express a deterministic relationship (the factor parameter IS the realization
/// of the linear combination of the factors).
#[derive(Debug, Clone)]
pub enum UnivariateFactor<D>
where D : Distribution
{

    /// Represents a stand-alone parametric distribution. The joint distribution
    /// p(y|theta) does not factor anymore.
    Empty,

    /// Represents a conditional expectation (deterministic link). The conditional distribution
    /// p(y|eta) does not factor into a pair of parametric distributions,
    /// but we have that eta = E[y|b] = x*b, where
    /// x is a constant vector of known realizations, and b ~ mnorm[p] is a prior or posterior
    /// for the linear combination coefficients x1 b1 + ... + xp bp. Since y is assumed iid.
    /// (conditional on b), var[y|b] is a fixed function of (x, b). We still evaluate only the
    /// probability of p(y|eta) for inference, but instead of using a constant eta, we use the
    /// realized samples of x to calculate for eta = x*b. The multivariate normality of b
    /// comes from the iid assumption, since p(y|eta)
    /// also factors as p(y1|eta_1)...p(yn|eta_n) during inference
    /// We can say that for the deterministic link we have this "sample factorization", while
    /// for the stochastic link (present at conjugate inference and multilevel models)
    /// we have sample and parameter factorization. The conjugate parameter theta is replaced by
    /// the natural parameter vector conditional upon which the sample is generated. If the conditional
    /// has a prior over the parameter vector, we evaluate the probability of this parameter vector wrt.
    /// the prior, but otherwise only the log-likelihood of the likelihood node is evaluated.
    /// Perhaps rename variant to Deterministic? (Deterministic links do not add a LL term: We only
    /// create a distribution that serves as a "proxy" for the parameter of the actual likelihood.
    CondExpect(MultiNormal),

    /// Represents a conjugate pair (stochastic link).
    /// The joint distribution p(y) factors as
    /// p(y|theta)p(theta). Perhaps rename variant to Stochastic?
    Conjugate(D)
}

fn univariate_log_prob<D>(
    y : DMatrixSlice<f64>,
    x : Option<DMatrixSlice<f64>>,
    factor : &UnivariateFactor<D>,
    eta : &DVector<f64>,
    log_part : &DVector<f64>, /*f64*/
    suf_factor : Option<DMatrix<f64>>
) -> f64
where D : Distribution
{
    // let eta = self.view_parameter(true);
    let eta_s = eta.rows(0, eta.nrows());
    // println!("eta = {}", eta_s);
    let factor_lp = match &factor {
        UnivariateFactor::Conjugate(d) => {
            assert!(y.ncols() == 1);
            assert!(x.is_none());
            let sf = suf_factor.unwrap();
            d.log_prob(sf.slice((0,0), sf.shape()), None)
        },
        UnivariateFactor::CondExpect(m) => {
            // If we are considering a conditional expectation factor, we consider not the
            // column sample vector, but a single row realization of a multivariate normal.
            // Assume here distribution is already scaled by the x argument we receive
            // (we shouldn't have to re-scale at every iteration).
            // let eta_t = DMatrix::from_rows(&[eta_s.clone_owned().transpose()]);
            // m.log_prob(eta_t.slice((0, 0), (1, eta_t.ncols())), x)

            // Just eval the log-probability of self, by changing parameters of factor.
            0.
        },
        UnivariateFactor::Empty => 0.
    };
    // eta_s.dot(&y.slice((0, 0), (y.nrows(), 1))) - lp + factor_lp
    (eta_s.component_mul(&y.slice((0, 0), (y.nrows(), 1))) - log_part).sum() + factor_lp
}

// TODO
// maybe differentiate Exponential and ScaledExponential
/// Generic trait shared by all exponential-family distributions. Encapsulate
/// all expressions necessary to build a log-probability with respect to a
/// sufficient statistic.
///
/// Exponential family members factor as:
/// a(theta)*y + b(eta) + c = eta*y + b(eta) + c
/// where a is a canonical-to-natural parameter mapping (link) function;
/// b is a function of the parameter alone (the log-partition), and c is
/// a base measure constant that can be ignored for the purpose of optimization.
pub trait ExponentialFamily<C>
where
    C : Dim,
    Self : Distribution
{

    /// Transforms a canonical parameter vector (i.e. parameter in the same scale
    /// as the maximum likelihood estimate) into a natural parameter vector; which
    /// is a continuous, unbounded transformation. The natural parameter is defined
    /// so that its inner product with a sufficient statistic completely defines
    /// the log-probability of the distribution with respect to the sample. This
    /// is a simple transformation of the location
    /// parameter for bounded distributions such as the Bernoulli; but involves a division
    /// by a separate parameter in unbounded distributions such as the Normal.
    fn link<S>(
        theta : &Matrix<f64, Dynamic, U1, S>
    ) -> Matrix<f64, Dynamic, U1, VecStorage<f64, Dynamic, U1>>
        where S : Storage<f64, Dynamic, U1>;

    /// Transforms a unbounded natural parameter into a bounded canonical parameter
    /// (the exact inverse of Self::link
    fn link_inverse<S>(
        eta : &Matrix<f64, Dynamic, U1, S>
    ) -> Matrix<f64, Dynamic, U1, VecStorage<f64, Dynamic, U1>>
        where S : Storage<f64, Dynamic, U1>;

    /// Collapses the independently distributed sample matrix y (with observations arranged over rows)
    /// into a low-dimensional sufficient statistic matrix.
    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64>;

    /// Calculates the log-probability with respect to the informed sufficient
    /// statistic matrix.
    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64;

    /// Normalization factor. Usually is a function of the
    /// dimensionality of the parameter, and is required for the distribution
    /// to integrate to unity. Function of a random sample. For univariate samples,
    /// this will be the element-specific base measure; For multivariate samples,
    /// that are evaluated against a single parameter; this will be a repeated value
    /// according to a given number of samples.
    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64>;

    // TODO change signature to
    // log_partition(&self)->f64 and just set internal (cached) log_partition
    // at the muting set_parameter method.
    /// The unnormalized log-probability is always defined as the inner product of a sufficient
    /// statistic with the natural parameter minus a term dependent on the parameter
    /// alone. This method updates this term for every parameter update.
    fn update_log_partition<'a>(&'a mut self, eta : DVectorSlice<'_, f64>);

    // The gradient of an exponential-family distribution is a linear function
    // of the sufficient statistic (constant) and the currently set natural parameter.
    // fn update_grad(&mut self, eta : DVectorSlice<'_, f64>);

    /// Retrieves the gradient of self, with respect to the currently set natural
    /// parameter and sufficient statistic. For univariate exponential family implementors,
    /// this is called the score function, because it is a generalization of the normal score
    /// (y - mu) / sigma. The output gradient has the same dimensionality as the sample y
    /// and the (expanded) natural parameter eta.
    fn grad(&self, y : DMatrixSlice<'_, f64>, x : Option<DMatrix<f64>>) -> DVector<f64> {
        match y.ncols() {
            1 => {
                // The same procedure will work for the scaled distributions (Normal and Multinormal)
                // except they will be divided by the standard deviation/pre-multiplied by the cholesky factor.
                // They will have specialized implementations for this method.
                let mut s = (y.column(0) - self.mean()).sum();

                DVector::from_element(1, s)
            },
            d => {
                // See specialized grad implementation at MultiNormal
                panic!("Unimplemented")
                /*let cov_inv = self.cov_inv().unwrap();
                assert!(cov_inv.nrows() == cov_inv.ncols());
                assert!(cov_inv.nrows() == self.mean().nrows());
                let yt = y.transpose();
                let yt_scaled = cov_inv.clone() * yt;
                let m_scaled = cov_inv * self.mean();
                let ys = yt_scaled.column_sum();
                yt_scaled - m_scaled*/
            }
        }
    }

    /// Function that captures the distribution invariances to
    /// location (first derivative) and scale (second derivative).
    /// This is a vector of the same size as the sample for univariate
    /// quantities, assuming the values according to the conditional expectation;
    /// but is a vector holding a single value for multivariate quantities,
    /// which are evaluated against a single parameter value.
    fn log_partition<'a>(&'a self) -> &'a DVector<f64>;

    /// Normalized probability of the independent sample y. Probabilities can only
    /// be evaluated directly if the distribution does not have any factors. If the
    /// distribution has any factors, only log_prob(.) (The unnormalized log-probability)
    /// can be evaluated. What we can do is calculate the KL divergence between a factor-free
    /// distribution A wrt. a factored distribution B by calling prob(.) over A and log_prob(.)
    /// over B, which is useful for variational inference.
    fn prob(&self, y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>) -> f64 {
        // TODO assert self does not have factors.
        // (Assume self does not have any factor,
        // since log_prob will evaluate whole graph)

        let mut unn_p = DVector::zeros(y.nrows());
        for (i, _) in y.row_iter().enumerate() {
            unn_p[i] = self.log_prob(y.rows(i,1), x).exp();
            //println!("lp = {}", unn_p[i]);
        }

        // (Moved base measure to addition at univariate_log_prob).
        let bm = Self::base_measure(y.clone());
        let p = bm.component_mul(&unn_p);
        //let p = unn_p;

        let joint_p = p.iter().fold(1., |jp, p| jp * p);
        joint_p
    }

    // TODO this method updates the sufficient statistic when self has a
    // conjugate factor. The conjugate factor always has its log-probability
    // evaluated wrt this element.
    //fn update_suff_stat(&mut self) {
    //}

    // The dispersion is mean() for the Poisson; 1. / mean() for the Bernoulli
    // and 1. / sigma for the normal. It is useful to express the score as the
    // error between the observation and the expected value times the dispersion.
    // The expected value is the first derivative of the log-partition; the variance
    // the second derivative times the dispersion.
    // fn dispersion()

}

type OptPosterior<'a> = Option<&'a mut (dyn Posterior + 'a)>;

/// FactorsMut is a safe iterator over mutable references to Posterior factors in a probabilistic graph.
/// It guarantees that you have either a reference to a current node or a pair of references to its
/// parents at any given iteration.
// pub struct Factors<'a>(Either<OptPosterior<'a>, (OptPosterior<'a>, OptPosterior<'a>)>); 
pub struct FactorsMut<'a> {
    curr : (OptPosterior<'a>, OptPosterior<'a>),
    returned_curr : bool
}

pub struct Factors<'a> {
    curr : Box<dyn Iterator<Item=&'a (dyn Posterior + 'a)>>
}

impl<'a> Iterator for FactorsMut<'a> {

    type Item = &'a mut dyn Posterior;
    
    fn next(&mut self) -> Option<&'a mut dyn Posterior> {
        match next_factor((self.curr.0.take(), self.curr.1.take()), &mut self.returned_curr) {
            Either::Left(Some(f)) => Some(f),
            Either::Right(parents) => {
                self.curr = parents;
                self.next()
            },
            Either::Left(None) => None,
        }
    }
    
}

fn next_factor<'a>(
    mut curr : (OptPosterior<'a>, OptPosterior<'a>), 
    returned_curr : &mut bool
) -> Either<OptPosterior<'a>, (OptPosterior<'a>, OptPosterior<'a>)> {
    if ! *returned_curr {
        *returned_curr = true;
        if curr.0.is_some() {
            Either::Left(curr.0)
        } else {
            if curr.1.is_some() {
                Either::Left(curr.1)
            } else {
                Either::Left(None)
            }
        }
    } else {
        if let Some(left) = curr.0.take() {
            *returned_curr = false;
            Either::Right(left.dyn_factors_mut())
        } else {
            if let Some(right) = curr.1.take() {
                *returned_curr = false;
                Either::Right(right.dyn_factors_mut())
            } else {
                Either::Right((None, None))
            }
        }
    }
}

impl<'a> Iterator for Factors<'a> {

    type Item = &'a mut dyn Posterior;
    
    fn next(&mut self) -> Option<&'a mut dyn Posterior> {
        unimplemented!()
    }
    
}

/// Implemented by distributions which can have their
/// log-probability evaluated with respect to a random sample directly.
/// Implemented by Normal, Poisson and Bernoulli. Other distributions
/// require that their log-probability be evaluated by using suf_log_prob(.)
/// by passing the sufficient statistic calculated from the sample.
/// Mathematically, a Likelihood node represents the best formulation about what a generative
/// process output should look like: Given a realization of its immediate prior, what range and
/// how concentrated the samples are around their expected value.
/// 
/// A full probabilistic graph is characterized by its outer factored
/// element, which is a distribution that implements the Likelihood trait. The
/// dimensionality of the data received by this element depends on how the graph
/// is built: If you feed a matrix of dimensionality n x m, the Likelihood element
/// take the first p columns (where p is the dimensionality of its random variable),
/// and dispatches the remaining m-p elements deeper into the graph, and the same process
/// is repeated by the remaining elements until the full data matrix is split and distributed
/// into the graph. If the element has a pre-set scale/shift factor (available only for
/// normal and multinormal nodes), the element will take the first p+q entries,
/// where q is the dimensionality of the scale matrix plus the dimensionality of the
/// shift vector. If the matrix cannot be split exactly between all graph elements,
/// the log_probability method will panic.
///
/// Likelihoods are factors that can live at the bottom of the probabilistic graph
/// and interact directly with data. If the model is composed of more than a single
/// likelihood distribution of the same type, use Factor<D>
/// (an iterator over distribution children).
///
/// Likelihood distributions (Normal, Poison, Bernoulli) have some extra information
/// that pure prior-posterior distributions do not (Gamma, Beta): They carry a condition enumeration
/// which tells how to interpret their natural parameter vector (is it a constant? a natural function?)
/// and they carry a Observation enumeration, which tell the observation status: Is it bound to a variable
/// name? If so, is this name related to a observation vector/matrix or only to a sufficient statistic?
///
/// The Likelihood trait offers functions that are useful before the inference procedure is run. The
/// counterpart of the likelihood for the posterior distribution is the predictive distribution, which
/// is also an "interfacing" distribution (is positioned at the top-level node of the graph). 
/// It offers a sample(.) and mean(.) methods, which allow to  retrieve relevant information from 
/// the Posterior Distribution. For conjugate inference, the Predicte is the same allocated object as
/// the prior, since there is no marginalization required for 1D distributions. For multivariate conjugate
/// and non-conjugate inference, usually a marginalization step is required to retrieve the posterior
/// distribution.
pub trait Likelihood
    where
        Self : Distribution //+ ?Sized,
        // C : Dim
{

    fn view_variables(&self) -> Option<Vec<String>>;
    
    /// Bind a sequence of variable names to this distribution. This causes calls to
    /// Likelihood::observe to bind to the respective variable names for any Sample implementor.
    fn with_variables(&mut self, vars : &[&str]) -> &mut Self where Self : Sized;
    
    /// Updates the full probabilistic graph from this likelihood node to all its
    /// parent factoring terms, binding any named likelihood nodes to the variable
    /// names found at sample. This incurs in copying the data from the sample implementor
    /// into a column-oriented data cache kept by each distribution separately.
    fn observe(&mut self, sample : &dyn Sample);
    // where
    //    R : IntoIterator<Item=&'a f64>,
    //    V : IntoIterator<Item=&'a f64>;
    
    /*/// General-purpose comparison of two fitted estimates, used for
    /// determining predictive accuracy, running cross-validation, etc.
    /// Comparisons can be made between two fitted models
    /// for purposes of hyperparameter tuning or model selection; between
    /// a fitted model and a saturated model for decision analysis; or between
    /// a fitted model and a null model for hypothesis testing.
    fn compare<'a>(&'a self, other : &'a Self) -> BayesFactor<'a, Self, Self>
        // where D : Distribution + Sized
        // where Self : Sized
    {
        BayesFactor::new(&self, &other)
    }*/

    /*/// Here, we approximate the relative entropy, or KL-divergence
    /// E[log_p(x) - log_q(x)] by the average of a few data point pairs (y, x)
    fn entropy<'a, D>(&'a self, other : &'a D, y : DMatrixSlice<'_, f64>, x : Option<DMatrixSlice<'_, f64>>) -> f64
        where D : Distribution + Sized
    {
        BayesFactor::new(&self, &other).log_diff(y, x)
    }*/

    // Returns the distribution with the parameters set to its
    // gaussian approximation (mean and standard error).
    // fn mle(y : DMatrixSlice<'_, f64>) -> Result<Self, anyhow::Error>;

    //{
    //    (Self::mean_mle(y), Self::se_mle(y))
    //}

    /*/// Returns a mean estimate using maximum likelihood estimation.
    fn mean_mle(y : DMatrixSlice<'_, f64>) -> f64;

    /// Returns a variance estimate using maximum likelihood estimation.
    fn var_mle(y : DMatrixSlice<'_, f64>) -> f64;

    /// Returns a standardized dispersion estimate for the maximum likelihood estimate of the mean.
    /// (standard error of mean). The standard deviation of the gaussian approximation
    /// to the posterior of conjugate models should approach this quantity as the sample
    /// size grows to be infinitely large.
    fn se_mle(y : DMatrixSlice<'_, f64>) -> f64 {
        let n = y.nrows() as f64;
        (Self::var_mle(y) / n).sqrt()
    }*/

    // Calls the closure for each distribution that composes the factored
    // joint distribution, in a depth-first fashion. Any Posterior distribution
    // can be a part of a nested bayesian inference problem, justifying this
    // visitor.
    // fn visit_factors<F>(&mut self, f : F) where F : Fn(&mut dyn Posterior);

    /// Access this distribution factor at the informed index. An collection of factors
    /// cannot be returned here because it would violate mutable exclusivity.
    /// Perhaps rename to prior_factors_mut()?
    fn factors_mut<'a>(&'a mut self) -> (Option<&'a mut dyn Posterior>, Option<&'a mut dyn Posterior>);
    
    // This would be nice if we could have Likelihood as trait objects.
    // fn likelihood_factors_mut<'a>(&'a mut self) -> (Option<&'a mut dyn Posterior>, Option<&'a mut dyn Posterior>);)
    
    // fn search_factor<'a>(&'a self, name : &str) -> Option<&Posterior> {
    // }
    
    fn iter_factors_mut<'a>(&'a mut self) -> FactorsMut<'a> 
    where
        Self : Sized
    {
        let post_parents = self.factors_mut();
        FactorsMut { curr : post_parents, returned_curr : false }
    }
    
    /*/// Returns a mutable iterator over this likelihood
    /// distribution factor(s).
    ///
    /// # Example
    ///
    /// ```
    /// use bayes::distr::*;
    ///
    /// let mut m = Normal::new(1,None,None).condition(Normal::new(1,None,None))
    ///     .condition(Gamma::new(1.,1.));
    /// m.factors_mut().visit::<_,()>(|f, _| println!("{}", f), None);
    /// ```
    fn factors_mut(&mut self) -> Factors;*/

    /*/// The conditional log-probability evaluation works because
    /// the sufficient statistic of Likelihood implementors is
    /// just the sum of the outcomes. Under this situation, the
    /// log-probability of the sufficient stat should equal the
    /// log-probability of the component-wise multiplication of
    /// the natural parameter with the individual samples.
    fn cond_log_prob(&self, y : DMatrixSlice<'_, f64>, x : Option<DMatrixSlice<'_, f64>>) -> f64 {
        match C::try_to_usize() {
            Some(_) => {
                let eta_cond = self.view_parameter(true);
                let log_part = self.log_partition();
                assert!(y.ncols() == eta_cond.ncols());
                assert!(log_part.nrows() == eta_cond.nrows() && log_part.nrows() == y.nrows());
                let mut lp = 0.0;
                let lp_iter = eta_cond.row_iter().zip(y.row_iter()).zip(log_part.iter());
                for ((e, y), l) in lp_iter {
                    lp += e.dot(&y) - l
                };
                lp
            },
            None => {
                self.suf_log_prob(y)
            }
        }
    }*/

    // Iterate over sister nodes if Factor; or returns a single distribution if
    // not a factor.
    // pub fn iter_sisters() -
}

/// The predictive distribution is the interfacing distribution of a probabilitic graph.
/// It is called a "prior predictive" before inference; and "posterior predictive" after fitting.
/// All distributions can generate samples by returning a matrix of values; but a predictive distribution
/// can additionally generate named samples. Those samples account for the inherent uncertainty of the
/// distribution and on the conditioning factors (prior or posteriors). All hand-built models are anchored
/// at a top-level distribution that implements Both Preditive and Likelihood. To retrieve the Posterior
/// predictions, the posterior object returned by your algorithm can also implement this trait. 
/// The "prior predictive" is also known as marginal likelihood, and gives the probability of observing
/// a data value by considering all variability of any priors it is conditioned over.
pub trait Predictive {

    /// Predicts a new data point after marginalizing over parameter values. 
    /// Requires that self.fit(.) has been called successfully at least once. You can make predictions conditional 
    /// on a new set of constant observations if you had fixed constants on the original model by passing Some(sample) to the
    /// argument, in which case the new sample will have the same dimensionality; or you can make
    /// the predictions based on the old fixed samples (if any) in which case the dimensionality of the
    /// predictions will follow the same dimensionality of the input data. A prediction returns always a mean
    /// (expected value) for all variables in the graph that were named (and are thus "likelihood" nodes, although
    /// their role here is as a Predictive distribution) and are not in the cond vector (if informed). 
    fn predict(&mut self, fixed : Option<&dyn Sample>) -> Box<dyn Sample>;
    
}

/// Used internally to sample all likelihood nodes together. The HashMap is then
/// Boxed into dyn Sample before being sent to the user. 
pub(crate) fn predict_from_likelihood<L>(
    lik : &mut L, 
    fixed : Option<&dyn Sample>
) -> HashMap<String, Vec<f64>> 
where
    L : Likelihood + ?Sized
{
    let names = lik.view_variables().unwrap();
    if let Some(fix) = fixed {
        lik.observe(fix);
    }
    let mut out : DMatrix<f64> = lik.sample();
    let mut sample : HashMap<String, Vec<f64>> = HashMap::new();
    for (i, name) in names.iter().enumerate() {
        sample.insert(name.to_string(), out.column(i).clone_owned().data.into());
    }
    sample
}

/*/// Return (mean, var) pair over a sample.
fn univariate_mle(y : DMatrixSlice<'_, f64>) -> (f64, f64) {
    assert!(y.ncols() == 1);
    let mle = y.iter().fold(0.0, |ys, y| {
        assert!(*y == 0. || *y == 1.); ys + y
    }) / (y.nrows() as f64);
    mle
}

fn var_mle(y : DMatrixSlice<'_, f64>) -> f64 {
    let m = Self::mean_mle(y);
    m * (1. - m)
}*/

/*
/// Priors represent a final or non-final generative process node. Their distinguishing characteristic
/// is that their log-likelihood is evaluated not with respect to a sample (as is the case for
/// Likelihood implementors) but with respect to a parameter of its immediate Likelihood node.
/// Mathematically, a prior formulation represents an agent's best guess on the different ways a generative process
/// works: What are the boundaries of the samples it generates, and how concentrated they are, or how different
/// parameters relate to each other. Prior formulations are especially important in high-dimensional problems, allowing
/// regularized solutions. Priors can represent a lower bound on the expected variance of the process for this objective,
/// an approach which require careful sensitivity analysis to be validated.
/// For example, to evaluate the log-probability of a Likelihood you do:
///
/// # (1) Evaluating the log-probability of a sample w.r.t. a likelihood 
/// let y = Normal::new(0.5, 0.2).variable("var1");
/// y.observe(&("var1", &[1,2,3]) as &dyn Sample);
/// let lik_lp = y.log_prob();
///
/// While to evaluate the log-probability of a prior node, you do:
///
/// # (2) Evaluating the log-probability of a sample w.r.t. a prior
/// let prior = Normal::new(0.1, 0.2);
/// let prior_lp = prior.log_prob(&y.view_parameter().as_slice());
/// let post_lp = lik_lp + post_lp;
///
/// The step above is performed lazily via the Conditioning trait: a.condition(b) means that
/// every time you evaluate the log-probability of the Likelihood node w.r.t. a &dyn Sample
/// the log-probability of (2) will be added to it.
/// Some distributions function as both Likelihoods and Priors, which mean they might be both the
/// first element and non-first element of probabilistic graphs (or in the middle of the graph
/// for multilevel models).
pub trait Prior {
    fn log_prob(&self, lik_param : &[f64]) -> f64;
}

*/
/// Posterior is a dynamic trait used by generic inference algorithms
/// to iterate over the probabilistic graph after the method cond_log_prob(.) is called
/// on its likelihood node. Inference algorithms should concern
/// only with whether the current distribution has either none, one or two factors
/// and the typical operations they might perform over them. Generic inference algorithms
/// are agnostic to which distributions compose the graph, but nevertheless always operate over them
/// in a valid way since elements can only be linked together via the compile-time checked Conditional<Target>
/// trait.
///
/// Posterior requires that Self implements Distribution, so all the conventional methods to mutate and
/// view parameter values are available, although parameters should always be treated on the natural scale.
/// sampling and calculation of log-probabilities are also available, and are probably the most useful.
///
/// Posterior also adds the set_approximation(.) and approximation(.) traits, which deal with
/// a gaussian approximation of self that can be accessed by the immediate child.
///
/// This trait is the way users can get posterior information from generic algorithms.
/// Algorithms will give back the received graph, but the approximation(.) and/or trajectory(.)
/// methods, instead of returning None, will return Some(MultiNormal) and/or Some(Trajectory)
/// users can query information from.
pub trait Posterior
    where Self : Debug + Display + Distribution
{

    /*fn dyn_factors(&mut self) -> (Option<&dyn Posterior>, Option<&dyn Posterior>) {
        let (fmut_a, fmut_b) = self.dyn_factors_mut();
        let f_a : Option<&dyn Posterior> = match fmut_a {
            Some(a) => Some(&(*a)),
            None
        };
        let f_b : Option<&dyn Posterior> = match fmut_b {
            Some(b) => Some(&(*b)),
            None
        };
        (f_a, f_b)
    }*/

    /*fn aggregate_factors<'a>(&'a mut self, factors : Factors<'a>) -> Factors<'a> {
        let (fmut_a, fmut_b) = self.dyn_factors_mut();
        let factors = if let Some(f_a) = fmut_a {
            factors.aggregate(f_a)
        } else {
            factors
        };
        let factors = if let Some(f_b) = fmut_b {
            factors.aggregate(f_b)
        } else {
            factors
        };
        factors
    }*/

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Posterior>, Option<&mut dyn Posterior>);

    fn iter_factors_mut<'a>(&'a mut self) -> FactorsMut<'a> 
        where Self: std::marker::Sized
    {
        FactorsMut { curr : (Some(self as &mut dyn Posterior), None), returned_curr : false }
    }
    
    /// Builds a predictive distribution from this Posterior by marginalization.
    /// Predictive distributions generate named samples, similar to the "prior predictive"
    /// (the likelihood). 
    fn predictive(&self) -> Box<dyn Predictive> {
        unimplemented!()
    }
    
    // fn iter_factors<'a>(&'a self) -> 
    
    /*fn iter_factors<'a>(&'a mut self) -> Box<dyn Iterator<Item=&'a mut dyn Posterior> + 'a> {
        let mut factors : Vec<&'a mut (dyn Posterior + 'a)> = Vec::new();
        let (opt_left, opt_right) = self.dyn_factors_mut();
        if let Some(left) = opt_left {
            factors.push(left);
            let lf = factors.last_mut().unwrap().iter_factors();
            factors.extend(lf);
        }
        if let Some(right) = opt_right {
            factors.push(right);
            let lr = factors.last_mut().unwrap().iter_factors();
            factors.extend(lr);
        }
        Box::new(factors.drain(0..))
    }*/
    
    /// Calls the closure for each distribution that composes the factored
    /// joint distribution, in a depth-first fashion.
    fn visit_post_factors(&mut self, f : &dyn Fn(&mut dyn Posterior)) {
        let (opt_lhs, opt_rhs) = self.dyn_factors_mut();
        if let Some(lhs) = opt_lhs {
            f(lhs);
            lhs.visit_post_factors(f);
        }
        if let Some(rhs) = opt_rhs {
            f(rhs);
            rhs.visit_post_factors(f);
        }
    }

    fn approximation_mut(&mut self) -> Option<&mut MultiNormal>;

    fn approximation(&self) -> Option<&MultiNormal>;

    fn start_trajectory(&mut self, size : usize);
    
    fn finish_trajectory(&mut self);
    
    fn trajectory(&self) -> Option<&Trajectory>;

    fn trajectory_mut(&mut self) -> Option<&mut Trajectory>;

    // Mark this variable fixed (e.g. at its current MLE) to avoid using it further as part of the
    // Inference algorithms by querying it via the fixed() method.
    // fn fix(&mut self)

    // Verify if this variable has been fixed by calling self.fix() at a previous iteration.
    // fn fixed(&self)

}

/// This trait represents an integration over remaining variables of the a probabilistic graph to
/// calculate the probability distribution of the variables identified by the names. 
pub trait Marginal<M> {

    fn marginal(&self, names : &[&str]) -> Option<M>;
    
}

/// There is a order of preference when retrieving natural parameters during
/// posterior estimation:
/// 1. If the distribution a started random walk started, get the parameter from its last step ; or else:
/// 2. If the distribution has an approximation set up, get the parameter from the approximation mean; or else:
/// 3. Get the parameter from the corresponding field of the implementor.
/// This order satisfies the typical strategy during MCMC of first finding a posterior mode approximation
/// and use that as a proposal.
fn get_posterior_eta<P>(post : &P) -> DVectorSlice<f64>
where P : Posterior
{
    match post.trajectory() {
        Some(rw) => rw.state(),
        None => {
            let param = match post.approximation() {
                Some(mn) => mn.view_parameter(true),
                None => post.view_parameter(true)
            };
            param.rows(0, param.nrows())
        }
    }
}

/// Updates the posterior internally-set parameter
/// using the random walk last step (if available)
/// or the approximation mean (if available).
fn update_posterior_eta<P>(post : &mut P)
    where P : Posterior
{
    let param = get_posterior_eta(&*post).clone_owned();
    post.set_parameter(param.rows(0, param.nrows()), true);
}

/*#[derive(Debug, Clone, Error)]
pub enum UnivariateError {

    #[error("Informed parameter value {0} outside distribution domain")]
    ParameterBounds(f64),

    #[error("Sample value {0} outside domain for distribution")]
    SampleDomain(f64)

}

#[derive(Debug, Clone, Error)]
pub enum CategoricalError {

}

#[derive(Debug, Clone, Error)]
pub enum MultivariateError {

}*/

/*/// Generic factor of distributions which have more than one factor
/// children of the same type. This structure implements Likelihood,
/// Distribution and Exponential generically by calling the methods of
/// its children, since factors are conditionally independent given
/// their parents. For distributions that have
/// a closed expression for the multivariate case (MultiNormal and Categorical)
/// the dedicated structures should be used.
pub struct Multi<D, P, C>
where
    D : Distrbution + ExponentialFamily<C> + Likelihood<C> + Conditional<P>,
    P : Distribution + ExponentialFamily<C> + Likelihood<C>
{
    prior : P,
    children : Vec<D>
}

impl<D, P, C> Multi<D, P, C>
where
    D : Distrbution + ExponentialFamily<C> + Likelihood<C> + Conditional<P>,
    P : Distribution + ExponentialFamily<C> + Likelihood<C>
{

    // Create a new distribution sequence with default parameters
    pub fn new(n : usize) -> Self {
        unimplemented!()
    }

    // pub fn iter() -> I;
    // pub fn iter_mut() -> I;

}*/

/// Enum carried by distributions which potentially function as likelihood nodes. Those distributions
/// might cache their observations or just a sufficient statistic of the observations. This field is
/// usually modified by Likelihood::observe. For multivariate normals, it can also be set by
/// MultiNormal::Fix.
enum Variate {

    /// This is the status when the distribution is created. To move to the Missing variant,
    /// call Likelihood::observe with a slice of variable names.
    Unspecified,
    
    /// Unobserved random variable. Can be a prior or hidden variable. Whether to move to the
    /// statistic or sample variant will depend on the algorithm: Conjugate methods might want to
    /// call self.set_statistic(.) While methods that need to preserve the full data vector
    /// might want to call self.set_sample(.).
    Missing{ vars : Vec<String> },
    
    /// An observed sample was compressed into a sufficient statistic and degrees of freedom (sample size).
    /// Much more efficient than storing the full sample, but the values cannot be re-used in a model
    /// with any conditionally-independent structure (the sample is assumed independent of the model formulation).
    Statistic{ vars : Vec<String>, value : DMatrix<f64>, dof : usize },
    
    /// Distribution yielded a full matrix of observations. Requires more memory, but is required if
    /// the sample is conditionally independent given the full model.
    Sample{ vars : Vec<String>, value : DMatrix<f64> }
}

/// Enum carried by distributions which potentially function as likelihood nodes. Those nodes might be
/// the top-level univariate or multivariate (mutually independent) leaves, observable "group" variables of
/// multilevel models, or hidden variables. Those nodes are special in that their parameter vector might
/// be interpreted in different ways depending on the conditioning strategy. Conditional models (e.g. GLMs)
/// do not form a fully probabilistic graph, but their factored representation also depend on constant terms
/// (which are not literally factors of the joint distribution, since the elements have been fixed and reduced
/// the dimensionality of the joint.
enum Condition {
    
    /// The parameter vector of this distribution is to be taken as is. This is the variant held by any
    /// likelihood node which is at the root of the graph, which happens early at model construction
    /// or if you are working with maximum likelihood methods.
    Constant,
    
    /// The parameter vector of this distribution is to be interpreted as coefficients to a linear
    /// combination of the columns of the fixed matrix (think about a regression model here). This
    /// problem aries naturally when you have a p-dimensional distribution and hold k of those factors
    /// fixed, effectively reducing the dimensionality of your joint. The linear formulation arises when
    /// a multivariate normal is held fixed as a function of a few parameters, while the non-fixed dimensions
    /// (mapping to the current distribution) are let free to vary.
    Deterministic{ fixed : DMatrix<f64> },
    
    Stochastic
    
}

/*fn observe_univariate<'a>(
    name : Option<String>, 
    n : usize,
    mut obs : Option<DVector<f64>>, 
    sample : &'a dyn Sample<'a>
) -> DVector<f64> {
    let mut obs = obs.take().unwrap_or(DVector::zeros(n));
    if let Some(name) = name {
        if let Some(col) = sample.variable(&name) {
            for (tgt, src) in obs.iter_mut().zip(col) {
                *tgt = *src;
            }
        }
    }
    obs
}*/

fn univariate_factor<'a, D>(factor : &'a mut UnivariateFactor<D>) -> Option<&'a mut dyn Posterior> 
where
    D : Distribution + Posterior
    // D : Conditional<P>,
    // P : Distribution
{
    match factor {
        UnivariateFactor::CondExpect(m) => {
            Some(m as &'a mut dyn Posterior)
        },
        UnivariateFactor::Conjugate(c) => {
            Some(c as &'a mut dyn Posterior)
        },
        UnivariateFactor::Empty => {
            // let no_distr = None;
            // Box::new(no_distr.iter().map(|d| *d))
            None
        }
    }
}


