use nalgebra::*;
use nalgebra::storage::*;
use std::fmt::Debug;
use std::ops::AddAssign;
use crate::model::decision::BayesFactor;
use std::fmt::Display;
use crate::fit::markov::Trajectory;
use anyhow;
use thiserror::Error;
use crate::sample::*;
use std::slice;
use either::Either;
use std::iter;
use std::collections::HashMap;
use std::iter::IntoIterator;
use crate::approx::*;

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

mod markov;

pub use markov::*;

pub use vonmises::*;

// pub mod mixture;

/*pub trait Univariate {

    fn mean(&self) -> f64;

    fn sample(&self) -> f64;

}

pub trait Multivariate {

    // Returns mean (column) vector.
    fn mean(&self) -> &[f64];

    // Index diagonal (ix, ix) of covariance matrix.
    fn var(&self, ix : usize) -> f64;

    // Returns covariance of the variable at the given informed index with all
    // others (index column of covariance matrxi).
    fn cov(&self, ix : usize) -> &[f64];

}*/

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

    fn set_natural<'a>(&'a mut self, eta : &'a mut dyn Iterator<Item=&'a f64>);
    
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

    // Returns a histogram if this distribution is univariate
    fn histogram(&self, bins : usize) -> Option<Histogram> {
        None
    }

    // Returns a kernel density estimate if this distribution is univariate
    fn smooth(&self, kernel : Kernel) -> Option<Density> {
        None
    }

    // Returns a symmetric interval around the mean if this distribution is univariate and symmetric
    fn zscore(&self, val : f64) -> Option<ZScore> {
        None
    }

    // Returns a highest-density interval if this distribution is univariate
    fn hdi(&self) -> Option<HDI> {
        None
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
    /// distribution parameter vector. TODO move to Likelihood
    fn joint_log_prob(&self /*, y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>*/ ) -> Option<f64>;

    /// Sample from the current distribution; If the sampling unit has multiple dimensions,
    /// they are represented over columns; If multiple units are sampled (if there are multiple
    /// entries for the parameter vector), a variable number of rows is emitted. Samples should
    /// follow the same structure as the argument to log_prob(.).
    /*fn sample(&self) -> DMatrix<f64> {
        let n = self.mean().nrows();
        let mut m : DMatrix<f64> = DMatrix::zeros(n, 1);
        self.sample_into((&mut m).into());
        m
    }*/

    fn sample(&self, dst : &mut [f64]);

    fn sample_into(&self, dst : DMatrixSliceMut<'_,f64>);
    
    fn param_len(&self) -> usize {
        self.view_parameter(true).nrows()
    }
    
    fn iter_factors_mut<'a>(&'a mut self) -> FactorsMut<'a>
        where Self: std::marker::Sized
    {
        FactorsMut { curr : (Some(self as &mut dyn Distribution), None), returned_curr : false }
    }

    fn dyn_factors(&self) -> (Option<&dyn Distribution>, Option<&dyn Distribution>) {
        unimplemented!()
    }

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Distribution>, Option<&mut dyn Distribution>) {
        unimplemented!()
    }

    // fn sample_into_transposed(&self, dst : DVectorSliceMut<'_, f64>);

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
    /// until the full graph is visited. TODO rename to modify_factor(.)
    fn factor_mut(&mut self) -> Option<&mut D>;

    // For MCMC inference, just call swap_factors starting from the leave nodes
    // iteratively until all likelihood factors are replaced by RandomWalks. All
    // distributions that implement Prior can be replaced by RandomWalk, which implements
    // posterior. Returns true when the factor was actually replaced. Does nothing if
    // distribution does not have a factor and returns false.
    fn swap_factor<E>(&mut self, other : E) -> bool
    where
        Self : Conditional<E>,
        E : Distribution
    {
        if let Some(f) = self.factor_mut() {
            *f = other;
            true
        } else {
            false
        }
    }
    
    //fn map_factor(&self, f : impl Fn(&D)) {
    //    self.view_factor().map(|factor| f(factor) )
    //}

    // Searches the given factor by variable name, returning it if successful.
    // fn search_factor(&self, name : &str) -> Option<&D>;

}

/// Maybe move to bayes::model, since this is related to model building.
/// Implemented by distributions which compose together to yield multivariate
/// joint distributions. Implementors are Normal(Normal)->MultiNormal and
/// MultiNormal(Normal)->MultiNormal for continuous variables; and Bernoulli->Bernoulli or
/// Categorical->Categorical for joint discrete distributions linked via conditional probability
/// tables (CPTs). This distribution is the basis to parse joint probabilistic models from JSON:
/// if two variable names are field names of a same JSON object, those names are assumed part of a 
/// joint distribution; the correlation or CPT argument define how those two elements will be linked.
/// TODO implement joint(normal, normal, type Corr=f64) to build bivariate relationships;
/// and also joint(normal, &[normal] type Corr=&[f64]) to build a convergent probabilistic dependency graph
/// RHS nodes to the LHS node.
/// Perhaps define Type Association. The association for a MultiNormal is a correlation slice holding correlation
/// of child with their parents; the association for a Markov is a conditional probability table
/// (slice of slices) of realization k to realization p.
/// The method association(&self) -> &[f64] retrieves the
/// association parameters from a child to their parents. Therefore, the user would access the covariance
/// matrix by either calling MultiNormal::cov or doing:
///
/// ```
/// let [n1, n2, n3] = (0..3).map(|_| Normal::new(1.0, 2.0)).collect();
///
/// // Build convergent dependency:
/// let m = { n1.joint((n2, n3), &[0.2, 0.3]);
///
/// // Build serial dependency:
/// let m = n1.joint(n2.joint(n3, &[0.2]), &[0.1]);
///
/// Build divergent dependency:
/// let m = (n2, n3).joint(n1, &[0.2, 0.3]); n3.joint(m) }
///
/// Default "canonical" bivariate convergent graph order is
/// established if MultiNormal is built in this way
/// let m = MultiNormal::new(&[1.0, 2.0, 3.0], cov);
///
/// By convention, the mean vector (and as a consequence the covariance matrix) are
/// ordered following the depth-first ordering of the network graph built.
///
/// // Retrieve correlations
/// m.depth_iter().skip(1).association();
///
/// // At a higher-level interface, we might keep a map of variable names to
/// // depth-first index; and a map of variable names to breadth-first index.
/// // To access a variable, the user might do m.depth_iter().nth(depth_vars["varname"])
/// ```

pub trait Joint<D>
where
    Self : Distribution + Sized,
    // Self : Distribution + Sized,
    D : Distribution,
    Self::Output : Distribution
{

    type Output;

    // To build a contingency table, the corr parameter should be replaced by
    // a probability mapping of the k-th child class to the i-th parent class.
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
/// is conditional on sampling the conjugate parameter; Fixed factors
/// express a deterministic relationship (the factor parameter IS the realization
/// of the linear combination of the factors).
#[derive(Debug, Clone)]
pub enum UnivariateFactor<D>
where 
    // D : Distribution + ExponentialFamily<Dynamic>
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
    Fixed(MultiNormal),

    /// Represents a conjugate pair (stochastic link).
    /// The joint distribution p(y) factors as
    /// p(y|theta)p(theta). Perhaps rename variant to Stochastic?
    Conjugate(D)
}

impl<D> UnivariateFactor<D> {

    pub fn fixed_obs(&self) -> Option<&DMatrix<f64>> {
        match &self {
            Self::Fixed(mn) => mn.fixed_observations(),
            _ => None
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self {
            UnivariateFactor::Empty => true,
            _ => false
        }
    }

}

/*pub enum MultivariateFactor {

    Empty,

    Conjugate(Box<MultiNormal>),

    Mixture(Categorical, Box<MultiNormal>),

    ScaledConjugate(Box<MultiNormal>, Wishart),

    ScaledMixture(Categorical, Box<MultiNormal>, Wishart)

}*/

/*impl<D> UnivariateFactor<D> {

    pub fn to_ref(&'a self) -> UnivariateFactor<'a D> {
        match self {
            UnivariateFactor::MultiNormal(mn) => UnivariateFactor::MultiNormal(mn.clone()),
            UnivariateFactor::Conjugate()
        }
        UnivariateFactor::Conjugate(&d)
    }
}*/

fn univariate_joint_log_prob<D>(
    y : Option<&DMatrix<f64>>,
    x : Option<&DMatrix<f64>>,
    factor : &UnivariateFactor<D>,
    eta : &DVector<f64>,
    log_part : &DVector<f64>,
    suf_factor : Option<DMatrix<f64>>
) -> Option<f64>
where 
    D : Distribution + ExponentialFamily<Dynamic>
{
    if let (Some(y), Some(x)) = (y, x) {
        assert!(x.nrows() == y.nrows());
    }
    if let Some(y) = y {
        assert!(y.nrows() == log_part.nrows());
        assert!(y.nrows() == eta.nrows());
    }
    let eta_s = eta.rows(0, eta.nrows());
    // println!("eta = {}", eta_s);
    let factor_lp = match &factor {
        UnivariateFactor::Conjugate(d) => {
            assert!(x.is_none());
            let sf = suf_factor?;
            
            // The posteior here is assumed to have n=1, so we evaluate its
            // log-probability with respect to the realization of the parameter
            // of the current likelihood node.
            d.suf_log_prob(sf.slice((0,0), sf.shape()))
        },
        UnivariateFactor::Fixed(m) => {
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
    let this_lp : f64 = eta.iter().zip(y?.iter()).zip(log_part.iter())
        .map(|((e, y), l)| (e * y) - l )
        .sum();
    Some(this_lp + factor_lp)
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
    /// statistic matrix. Perhaps move this to Prior trait? So the evaluation
    /// wrt. a single parameter value vector is typical of Gamma, Normal (for n=1), 
    /// MultiNormal (for n=1 as well) and Beta.
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
    fn update_log_partition<'a>(&'a mut self, /*eta : DVectorSlice<'_, f64>*/ );

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
    /// over B, which is useful for variational inference. TODO must have distribution-specific
    /// implementation, since this is just a element access for discrete distributions Bernoulli
    /// and categorical; or a call to erf(x) for Normal/MultiNormal.
    fn prob(&self) -> Option<f64> {
        // TODO assert self does not have factors.
        // (Assume self does not have any factor,
        // since log_prob will evaluate whole graph)

        /*let mut unn_p = DVector::zeros(y.nrows());
        for (i, _) in y.row_iter().enumerate() {
            unn_p[i] = self.joint_log_prob().unwrap().exp();
            //println!("lp = {}", unn_p[i]);
        }

        // (Moved base measure to addition at univariate_log_prob).
        let bm = Self::base_measure(y.clone());
        let p = bm.component_mul(&unn_p);
        //let p = unn_p;

        let joint_p = p.iter().fold(1., |jp, p| jp * p);
        Some(joint_p)*/
        unimplemented!()
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

type OptDistribution<'a> = Option<&'a mut (dyn Distribution + 'a)>;

/// FactorsMut is a safe iterator over mutable references to distribution factors in a probabilistic graph.
/// It guarantees that you have either a reference to a current node or a pair of references to its
/// parents at any given iteration.
// pub struct Factors<'a>(Either<OptDistribution<'a>, (OptDistribution<'a>, OptDistribution<'a>)>);
pub struct FactorsMut<'a> {
    curr : (OptDistribution<'a>, OptDistribution<'a>),
    returned_curr : bool
}

pub struct Factors<'a> {
    curr : Box<dyn Iterator<Item=&'a (dyn Distribution + 'a)>>
}

impl<'a> Iterator for FactorsMut<'a> {

    type Item = &'a mut dyn Distribution;
    
    fn next(&mut self) -> Option<&'a mut dyn Distribution> {
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
    mut curr : (OptDistribution<'a>, OptDistribution<'a>),
    returned_curr : &mut bool
) -> Either<OptDistribution<'a>, (OptDistribution<'a>, OptDistribution<'a>)> {
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

    type Item = &'a mut dyn Distribution;
    
    fn next(&mut self) -> Option<&'a mut dyn Distribution> {
        unimplemented!()
    }
    
}

/// Implemented by distributions which are non-final graph nodes. Prior distributions
/// have a dimensionality of one, and only receive parameter values.
pub trait Prior {

    type Parameter;

    fn prior(param : Self::Parameter) -> Self;

}

/*
Proposal: Reduce likelihood into:

pub trait Likelihood<S> {

    fn observe(&mut self, impl Iterator<Item=&S>);

    fn likelihood(impl Iterator<Item=&S>) -> Self;
}

So that we might have one likelihood for each possible Sample type T. We can provide
a generic impl Likelihood<&dyn AsRef[f64]> for MultiNormal or impl Likelihood<&dyn Iterator<Item=&f64>>.
If the user implements those traits for its custom type, then &[T] where T is any user-defined type
can be used as samples to feed a distribution.

To deal with dynamic structures (Row, Value), we can do:

impl Likelihood<(&Value, String)> for Normal { }

Where to use the second entry to perform a custom indexing operation. Or even

impl Likelihood<(&dyn Index<&str>, String)> for Normal { }

To allow for any string-indexable types to yield observations.
*/

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
///
/// Likelihood distributions might have a set of fixed variables, which require a multinormal conditioning
/// factor representing linear combination coefficients for sampling and log-probability calculations. The
/// linearity of those factors arises by assuming that the natural parameter of the current distribution
/// and the fixed factors follow a joint multivariante distribution; which was conditioned on the fixed
/// factors to yield the current mean estimate. In a factoring graph, the observations themselves do not
/// constitute an independent factor, since they are fixed and do not have their own log-likelihood to
/// be considered.
pub trait Likelihood<O>
    where
        Self : Distribution, //+ ?Sized,
        O : ?Sized
        // C : Dim
{

    fn sample_size(&self) -> usize {
        self.view_variable_values().map(|vars| vars.nrows() ).unwrap_or(0)
    }

    fn likelihood<'a>(obs : impl IntoIterator<Item=&'a O>) -> Self
    where
        O : 'a,
        Self : Sized
    {
        unimplemented!()
    }

    fn observe<'a>(&mut self, obs : impl IntoIterator<Item=&'a O>)
    where
        O : 'a
    {
        unimplemented!()
    }

    fn observe_owned(&mut self, obs: impl IntoIterator<Item=O>) {
        unimplemented!()
    }

    fn view_variables(&self) -> Option<Vec<String>>;
    
    fn view_fixed(&self) -> Option<Vec<String>>;
    
    fn view_variable_values(&self) -> Option<&DMatrix<f64>>;

    fn view_fixed_values(&self) -> Option<&DMatrix<f64>> {
        unimplemented!()
    }

    /// Bind a sequence of variable names to this distribution. This causes calls to
    /// Likelihood::observe to bind to the respective variable names for any Sample implementor.
    fn with_variables(&mut self, vars : &[&str]) -> &mut Self where Self : Sized;
    
    /// Bind a sequence of variable names which are assumed to be fixed with respect to this
    /// distribution. This causes calls to Likelihood::observe to bind the respective data.
    /// Fixed variable names 
    fn with_fixed(&mut self, fixed : &[&str]) -> &mut Self where Self : Sized;
    
    /// Updates the full probabilistic graph from this likelihood node to all its
    /// parent factoring terms, binding any named likelihood nodes to the variable
    /// names found at sample. This incurs in copying the data from the sample implementor
    /// into a column-oriented data cache kept by each distribution separately. A full probabilistic
    /// tree contains at least one top-level likelihood node, but possibly many likelihood nodes up to its
    /// kth level, corresponding to multilevel models, where observations are conditioned on other observations.
    /// After the k+1 level, there cannot be distributions which receive data (only non-named prior distributions).
    fn observe_sample(&mut self, sample : &dyn Sample, vars : &[&str]);

    /* pub trait Process<I> {

        fn initialize(&self) -> Option<I>;
        fn innovation(&self) -> Option<I>;
    }

    impl Iterator<Item=f64> for MyType { }
    impl markov::Process for MyType { }

    fn observe_process(&mut self, process : &dyn Process);
    */

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

    // Access this distribution factor at the informed index. An collection of factors
    // cannot be returned here because it would violate mutable exclusivity.
    // Perhaps rename to prior_factors_mut()?
    /*fn factors_mut<'a>(&'a mut self) -> (Option<&'a mut dyn Posterior>, Option<&'a mut dyn Posterior>) {
        unimplemented!()
    }*/
    
    // This would be nice if we could have Likelihood as trait objects.
    // fn likelihood_factors_mut<'a>(&'a mut self) -> (Option<&'a mut dyn Posterior>, Option<&'a mut dyn Posterior>);)
    
    // fn search_factor<'a>(&'a self, name : &str) -> Option<&Posterior> {
    // }
    
    // TODO move to conditional<T>
    /*fn iter_factors_mut<'a>(&'a mut self) -> FactorsMut<'a>
    where
        Self : Sized
    {
        let post_parents = self.factors_mut();
        FactorsMut { curr : post_parents, returned_curr : false }
    }*/
    
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

    /*Need to figure out a way to update the data of a likelihood that satisfies the Markov
    property (only need to keep the last state to calculate the current log-prob).

    Perhaps
    distr::markov(1) creates an order-1 markov process (any calls to observe will overwrite
    a single data point). distr::markov(n > 1) will store elements in a VecDequeue (ringbuffer)
    which is made contiguous whenever the log-probability needs to be calculated and erases any
    last old element whenever a new element is observed, preserving n elements at all times.*/
}

/// Held by likelihood/predictive implementors.
pub struct Data<T> {
    observed : Option<Vec<T>>,
    predicted : Option<Vec<T>>
}

/*/*
TODO: Perhaps change trait to:
fn predict(&'a mut self) -> &'a dyn Sample;
fn view_predictions(&'a self) -> Option<&'a dyn Sample>;
Which will avoid a heap allocation when making new predicitons (can just re-use the 
data buffer).
*/ 
/// The predictive distribution is the interfacing distribution of a probabilitic graph.
/// It is called a "prior predictive" before inference; and "posterior predictive" after fitting.
/// All distributions can generate samples by returning a matrix of values; but a predictive distribution
/// can additionally generate named samples. Those samples account for the inherent uncertainty of the
/// distribution and on the conditioning factors (prior or posteriors). All hand-built models are anchored
/// at a top-level distribution that implements Both Preditive and Likelihood. To retrieve the Posterior
/// predictions, the posterior object returned by your algorithm can also implement this trait. 
/// The "prior predictive" is also known as marginal likelihood, and gives the probability of observing
/// a data value by considering all variability of any priors it is conditioned over.
/// This trait can simply be implemented by the same distributions that implement Likelihood. If they still have
/// the prior nodes, it returns None; If they have the typical posterior nodes ImportanceDraw<D> or Trajectory<D>
/// they return Some(sample).
pub trait Predictive {

    /// Predicts a new data point after marginalizing over parameter values. 
    /// Requires that self.fit(.) has been called successfully at least once. You can make predictions conditional 
    /// on a new set of constant observations if you had fixed constants on the original model by passing Some(sample) to the
    /// argument, in which case the new sample will have the same dimensionality; or you can make
    /// the predictions based on the old fixed samples (if any) in which case the dimensionality of the
    /// predictions will follow the same dimensionality of the input data. A prediction returns always a mean
    /// (expected value) for all variables in the graph that were named (and are thus "likelihood" nodes, although
    /// their role here is as a Predictive distribution) and are not in the cond vector (if informed). 
    fn predict<'a>(&'a mut self, fixed : Option<&dyn Sample>) -> Option<&'a dyn Sample>;
    
    fn view_prediction<'a>(&'a self) -> Option<&'a dyn Sample>;
    
}*/

/// Predictive is implemented by all distributions which support ancestral sampling.
/// While distr.sample() will generate samples only using information local to the distribution
/// (its immediate location and scale parameters), the distr.predict(.) will evolve the RNG for
/// each node in the factor graph and re-sample the value using the currently-set parameter
/// values. Predict can also be used for fixed distributions, since predicting a value for
/// original fixed predictors is done by calling observe_fixed(.) over the fixed factors and then
/// calling predict(.) on the child node.
pub trait Predictive<O> {

    fn predict(&self) -> O;

}

/*/// Used internally to sample all likelihood nodes together. The HashMap is then
/// Boxed into dyn Sample before being sent to the user. 
pub(crate) fn predict_from_likelihood<L, O>(
    lik : &mut L, 
    fixed : Option<&dyn Sample>
) -> HashMap<String, Vec<f64>> 
where
    L : Likelihood<O> + ?Sized
{
    let names = lik.view_variables().unwrap();
    if let Some(fix) = fixed {
        lik.observe_sample(fix);
    }
    let mut out : DMatrix<f64> = lik.sample();
    let mut sample : HashMap<String, Vec<f64>> = HashMap::new();
    for (i, name) in names.iter().enumerate() {
        sample.insert(name.to_string(), out.column(i).clone_owned().data.into());
    }
    sample
}*/

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
///
/// The distinct characteristic of a prior is that its log-probability can be evaluated with respect
/// to a constant. Therefore, Distributions that implement prior can be initialized by receiving
/// constant values:
///
/// let g = Gamma::prior(0.1, 1);
/// let n = Normal::prior(0.5, 0.2);
///
/// Which is distinct from likelihood, which are initialized by receiving a sample:
///
/// let n = Normal::likelihood(&[1,2,3]);
/// let b = Bernoulli::likelihood(&[true, false, true]);
///
/// Or modified by receiving another sample:
/// b.observe(&[true, true, false]);
///
/// Likelihood<T> means that the distribution can be instantiated by receiving a sample
/// of type T:
/// impl Likelihood<f64> for Normal; impl Likelihood<f32> for Normal; impl Likelihood<bool> for Bernoulli.
/// impl Likelihood<&[f64]> for MultiNormal.
///
/// Conditional regression-like distributions are built by conditioning on constants:
///
/// let y = [1.0, 2.0];
/// let x = [4.0, 5.0];
/// let n = Normal::likelihood(&y).condition(Constant::new(&[x]));
///
/// The simplified traits will then be:
/// pub trait Prior{
///    fn prior(&[f64]) -> Self;
///    fn fix(&mut self, &[f64]); /* Receives a constant parameter vector */
/// }
///
/// pub trait Likelihood<T> {
///     fn likelihood(&impl IntoIterator<T>)->Self;
///     fn observe(&mut self, &impl IntoIterator<T>);
/// }
///
/// pub trait Sample {
///     fn bind(&self, prob : &mut impl Index<&str, Output = &mut dyn Likelihood>);
/// }
/// fn observe_sample(&mut self, &dyn Sample);
/// By using IntoIterator, we can pass both closures and slices.
/*
To reference distributions in the data table at high-level applications, we can then have:
pub enum Distribution {
    Bern(&Bernoulli),
    Norm(&Norm),
    Const(&Constant)
    ...
}

/// Containts a key-value map of distributions accessible by name.
pub struct Model {
    content : HashMap<String, Distribution>
}

impl Model<D> {
    fn observe_all(&mut self, HashMap<&str, &Column[f64]);
}

Then we have that Distributions serialize into simple objects like:
{ "mean" : 0.2, "var" : 0.1 }
{ "prob" : 0.1 }
{ "mean" : 0.1, "var" : 0.1, "obs": [1.0, 2.0, 3.0] }

But Model serialize into key-value maps of name to those objects:
{ "age" : { mean :  { "date" : { "mean" : 0.1, "var" : 0.2 } }, var : 0.2 }
*/

/*/// Posterior is a dynamic trait used by generic inference algorithms
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
/// The API of posterior can have just posterior_factors(.), which yield the typical posterior distribution
/// factors: ImportanceDraw<D> for posterior implementor Importance with prior node D; or Trajectory<D> for
/// posterior implementor RandomWalk with prior node D.
pub trait Posterior
    where Self : Debug + Display + Distribution + Markov
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

    // fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Posterior>, Option<&mut dyn Posterior>) {
    //    unimplemented!()
    // }

    /*/// Builds a predictive distribution from this Posterior by marginalization.
    /// Predictive distributions generate named samples, similar to the "prior predictive"
    /// (the likelihood). 
    fn predictive(&self) -> Box<dyn Predictive> {
        unimplemented!()
    }*/
    
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
        /*let (opt_lhs, opt_rhs) = self.dyn_factors_mut();
        if let Some(lhs) = opt_lhs {
            f(lhs);
            lhs.visit_post_factors(f);
        }
        if let Some(rhs) = opt_rhs {
            f(rhs);
            rhs.visit_post_factors(f);
        }*/
        unimplemented!()
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

}*/

/// This trait represents an integration over remaining variables of the a probabilistic graph to
/// calculate the probability distribution of the variables identified by the names. 
pub trait Posterior<M> {

    // marginal is not really an operation to be performed on posteriors only; it just
    // happens that the implementors (normal/multinormal/randomwalk) all serve as posteriors
    // at some point. A more "exclusive" operation for analytical posteriors would be to yield a
    // pseudo-data count (Gamma and Beta); or the sample size count that was used to fit the distribution.
    // It would return None if the distribution is a prior; and Some(count) if the distribution was
    // built by sampling n values. Same as Likelihood::sample_size(.). Perhaps move marginal(.) to a
    // separate Marginal trait.
    fn marginal(&self, ix : usize) -> Option<M>;
    
}

/*/// There is a order of preference when retrieving natural parameters during
/// posterior estimation:
/// 1. If the distribution a started random walk started, get the parameter from its last step ; or else:
/// 2. If the distribution has an approximation set up, get the parameter from the approximation mean; or else:
/// 3. Get the parameter from the corresponding field of the implementor.
/// This order satisfies the typical strategy during MCMC of first finding a posterior mode approximation
/// and use that as a proposal.
fn get_posterior_eta<P>(post : &P) -> DMatrixSlice<f64>
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
    post.natural_mut().copy_from(param.rows(0, param.nrows()));
}*/

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

fn sample_conjugate<D>(distr : &D, n : usize) -> DVector<f64>
where
    D : Distribution
{
    /*let theta_draw = distr.sample().row(0).clone_owned().transpose();
    let theta = DVector::from_element(n, theta_draw[0]);
    theta*/
    unimplemented!()
}

fn sample_cond_expect(x : &DMatrix<f64>, mn : &MultiNormal) -> DVector<f64> {
    /*let beta = mn.sample().row(0).clone_owned().transpose();
    let eta = x.clone_owned() * beta;
    eta*/
    unimplemented!()
}

/// If this distribution has a Conjugate OR Fixed variant, resolve the sampling
/// of the parent into a vector. If not, return None. The returned vector is always a
/// natural parameter.
pub fn sample_natural_factor<D>(
    fixed : Option<&DMatrix<f64>>,
    factor : &UnivariateFactor<D>,
    n : usize
) -> Option<DVector<f64>>
where
    D : Distribution
{
    match (fixed, factor) {
        (_, UnivariateFactor::Conjugate(d)) => Some(sample_conjugate(d, n)),
        (Some(x), UnivariateFactor::Fixed(mn)) => Some(sample_cond_expect(x, mn)),
        _ => None
    }
}

/// Sample from a location factor, multiply by fixed values, and then apply the inverse-link to get
/// the canonical parameter values.
pub fn sample_canonical_factor<E, D>(
    fixed : Option<&DMatrix<f64>>,
    factor : &UnivariateFactor<D>,
    n : usize
) -> Option<DVector<f64>>
where
    E : ExponentialFamily<U1>,
    D : Distribution
{
    sample_natural_factor(fixed, factor, n).map(|eta| E::link_inverse(&eta) )
}

pub fn sample_natural_factor_boxed<D>(
    fixed : Option<&DMatrix<f64>>,
    factor : &UnivariateFactor<Box<D>>,
    n : usize
) -> Option<DVector<f64>>
where
    D : Distribution
{
    match (fixed, factor) {
        (_, UnivariateFactor::Conjugate(boxed_d)) => Some(sample_conjugate(boxed_d.as_ref(), n)),
        (Some(ref x), UnivariateFactor::Fixed(ref mn)) => Some(sample_cond_expect(x, mn)),
        _ => None
    }
}

/// Trait implemented by distributions which can be conditioned on a fixed
/// multinormal.
pub trait FixedConditional {

    fn fixed_factor<'a>(&'a mut self) -> Option<&'a mut MultiNormal>;

}

/// Private trait implemented by distributions which yield observations. This allows
/// generic implementations of the likelihood(.) and observe(.) methods. The private
/// trait pattern is useful when we have several structures that share some common
/// fields. This is somewhat like OOP inheritance: We have some generic function
/// that make use of specific implementations.
pub trait Observable {

    fn observations(&mut self) -> &mut Option<DMatrix<f64>>;

    fn sample_size(&mut self) -> &mut usize;

}

fn observe_univariate_generic<'a, T>(lik : &mut impl Observable, obs : impl Iterator<Item=&'a T>)
where
    f64 : From<T>,
    T : Copy + 'a
{
    let mut v : Vec<f64> = Vec::new();
    v.extend(obs.into_iter().map(|el| f64::from(*el) ));
    let n = v.len();
    *lik.observations() = Some(DMatrix::from_vec(n, 1, v));
    *lik.sample_size() = n;
}

fn observe_multivariate_generic<'a, T>(lik : &mut impl Observable, obs : impl Iterator<Item=&'a [T]>)
where
    f64 : From<T>,
    T : Copy + 'a
{
    let mut v : Vec<f64> = Vec::new();

    let mut dim = 0;

    // Create row-wise (wide) observation matrix
    for (i, row) in obs.into_iter().enumerate() {
        v.extend(row.into_iter().map(|el| f64::from(*el) ));

        // Guarantees slices at 1..n have the same size as slice[0]
        if i == 0 {
            dim = v.len();
        } else {
            assert!(row.len() == dim);
        }
    }
    let n = v.len() / dim;
    let mut obs = DMatrix::from_vec(n, dim, v);

    // Transform to tall observation matrix
    // obs = obs.transpose();

    *lik.observations() = Some(obs);
    *lik.sample_size() = n;
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

fn univariate_factor<'a, D>(factor : &'a UnivariateFactor<D>) -> Option<&'a dyn Distribution>
where
    D : Distribution + /*Posterior +*/ ExponentialFamily<Dynamic>
{
    match factor {
        UnivariateFactor::Fixed(m) => {
            Some(m as &'a dyn Distribution)
        },
        UnivariateFactor::Conjugate(c) => {
            Some(c as &'a dyn Distribution)
        },
        UnivariateFactor::Empty => {
            // let no_distr = None;
            // Box::new(no_distr.iter().map(|d| *d))
            None
        }
    }
}

fn univariate_factor_mut<'a, D>(factor : &'a mut UnivariateFactor<D>) -> Option<&'a mut dyn Distribution>
where
    D : Distribution /*+ Posterior*/ + ExponentialFamily<Dynamic>
    // D : Conditional<P>,
    // P : Distribution
{
    match factor {
        UnivariateFactor::Fixed(m) => {
            Some(m as &'a mut dyn Distribution)
        },
        UnivariateFactor::Conjugate(c) => {
            Some(c as &'a mut dyn Distribution)
        },
        UnivariateFactor::Empty => {
            // let no_distr = None;
            // Box::new(no_distr.iter().map(|d| *d))
            None
        }
    }
}

fn observe_real_columns(names : &[&str], sample : &dyn Sample, opt_obs : &mut Option<DMatrix<f64>>, n : usize) {
    if opt_obs.is_none() {
        *opt_obs = Some(DMatrix::zeros(n, names.len() + 1));
    }
    let obs = opt_obs.as_mut().unwrap();

    // Add intercept term
    obs.column_mut(0).iter_mut().for_each(|item| *item = 1.0 );

    // Add data to non-intercept columns
    for (i, name) in names.iter().cloned().enumerate() {
        if let Variable::Real(col) = sample.variable(&name) {
            let mut total = 0;
            for (tgt, src) in obs.column_mut(i + 1).iter_mut().zip(col) {
                *tgt = src;
                total += 1;
            }
            assert!(total == n);
        }
    }
}

/// Verify if no variables are missing
fn verify_complete(sample : &dyn Sample, var : &str) -> bool {
    match sample.variable(var) {
        Variable::Missing => false,
        _ => true
    }
}

/// Guarantee all data columns of sample are present and match valid variables. Note that it is
/// not possible to verify if sample has any "extra" valid names using just the sample API (which
/// does not list the available names; just return variables for any requested name).
fn verify_data_completeness<L, O>(distr : &L, sample : &dyn Sample) -> bool
where
    L : Likelihood<O> + Distribution
{
    // Fixed is not required to be present
    let fixed_complete = distr.view_fixed()
        .map(|vars| vars.iter().all(|var| verify_complete(sample, var)) )
        .unwrap_or(true);

    // All random variables should be required
    let rand_complete = distr.view_variables()
        .map(|vars| vars.iter().all(|var| verify_complete(sample, var )))
        .unwrap_or(false);
    fixed_complete && rand_complete
}

/// Verify if the distribution name is present in the sample. Returning a false
/// result is important if only the fixed values should be informed for a regression problem.
fn verify_if_name_in_sample<L, O>(distr : &L, sample : &dyn Sample) -> bool
where
    L : Likelihood<O> + Distribution
{
    if let Some(vars) = distr.view_variables() {
        for var in vars.iter() {
            match sample.variable(var) {
                Variable::Missing => { },
                _ => return true
            }
        }
        false
    } else {
        false
    }
}

/// Verify if this likelihood has fixed values associated with it
/// (i.e. if it is a likelihood conditioned on constants)
fn has_fixed<L, O>(distr : &L) -> bool
where
    L : Likelihood<O> + Distribution
{
    distr.view_fixed_values().is_some() && distr.view_fixed().is_some()
}

/// Builds a regression prediction matrix, assuming distr has fixed values. The fixed
/// variables matrix should have p+1 columns, with the 1 standing for the constant intercept,
/// although the user does not access this value via the Sample implementation. The observations
/// are appended to the last column.
fn build_regression_predictions<L, O>(distr : &L) -> (Vec<String>, DMatrix<f64>)
where
    L : Likelihood<O> + Distribution
{
    let x = distr.view_fixed_values().as_ref().unwrap().clone_owned();

    // Append random observations to last column of fixed values
    let ncols = x.ncols();
    let mut data = x.insert_column(ncols, 0.0);
    distr.sample_into(data.slice_mut((0, data.ncols() - 1), (data.nrows(), 1)));

    // Append names in the same order
    let mut obs_names : Vec<String> = Vec::new();
    obs_names.extend(distr.view_fixed().as_ref().unwrap().iter().cloned());
    obs_names.extend(distr.view_variables().clone().unwrap().iter().cloned());

    assert!(obs_names.len() == data.ncols());

    (obs_names, data)
}

fn build_generalized_regression_predictions<L, O, F, T>(distr : &L, transf : F) -> HashMap<String, Either<Vec<f64>, Vec<T>>>
where
    F : Fn(&f64)->T,
    L : Likelihood<O> + Distribution
{
    let (names, regr_preds) = build_regression_predictions(distr);
    let ncols = regr_preds.ncols();
    let mut preds = HashMap::new();
    for i in 0..(ncols - 1) {
        let x_col_data : Vec<f64> = regr_preds.column(i).iter().cloned().collect();
        preds.insert(names[i].clone(), Either::Left(x_col_data));
    }
    let y_col_data : Vec<T> = regr_preds.column(ncols - 1).as_slice().iter().map(transf).collect();
    preds.insert(names[ncols - 1].clone(), Either::Right(y_col_data));
    preds
}

/// Builds a prediction 1-column matrix, assuming distr has no conditional fixed values
/// (i.e. is conditioned on a constant vector).
fn build_constant_predictions<L, O>(distr : &L) -> (Vec<String>, DMatrix<f64>)
where
    L : Likelihood<O> + Distribution
{
    /*let mut obs_names : Vec<String> = Vec::new();
    obs_names.extend(distr.view_variables().clone().unwrap().iter().cloned());
    let samples = distr.sample();
    (obs_names, samples)*/
    unimplemented!()
}

fn build_generalized_constant_predictions<L, O, F, T>(distr : &L, transf : F) -> HashMap<String, Either<Vec<f64>, Vec<T>>>
where
    F : Fn(&f64)->T,
    L : Likelihood<O> + Distribution
{
    /*// Collect name
    let mut obs_names : Vec<String> = Vec::new();
    obs_names.extend(distr.view_variables().clone().unwrap().iter().cloned());

    // Collect samples
    let samples = distr.sample();
    let transf_obs : Vec<T> = samples.as_slice().iter().map(transf).collect();

    assert!(obs_names.len() == samples.ncols() && obs_names.len() == 1);

    let mut obs_hash = HashMap::new();
    obs_hash.insert(obs_names.remove(0), Either::Right(transf_obs));
    obs_hash*/
    unimplemented!()
}

fn try_build_real_predictions<L, O>(distr : &L) -> Result<(Vec<String>, DMatrix<f64>), String>
where
    L : Likelihood<O> + Distribution
{
    if has_fixed(distr) {
        Ok(build_regression_predictions(distr))
    } else {
        Ok(build_constant_predictions(distr))
    }
}

fn try_build_generalized_predictions<L, O, F, T>(distr : &L, transf : F) -> Result<HashMap<String, Either<Vec<f64>, Vec<T>>> , String>
where
    F : Fn(&f64)->T,
    L : Likelihood<O> + Distribution
{
    if has_fixed(distr) {
        Ok(build_generalized_regression_predictions(distr, transf))
    } else {
        Ok(build_generalized_constant_predictions(distr, transf))
    }
}

/*fn collect_fixed_if_required<L,O>(distr : &mut L, informed_fixed : Option<&dyn Sample>) -> Result<(), String>
where
    L : Likelihood<O> + Distribution
{
    match (has_fixed(distr), informed_fixed) {
        (true, Some(fixed)) => {
            if !verify_data_completeness(distr, fixed) {
                return Err(String::from("Incomplete data"));
            }
            if verify_if_name_in_sample(distr, fixed) {
                return Err(String::from("Variable name cannot be in new informed fixed sample"));
            }
            distr.observe_sample(fixed);
            Ok(())
        },
        (false, Some(_)) => {
            Err(String::from("Informed fixed value for constant-parameter distribution"))
        },
        _ => Ok(())
    }
}*/

/*/// Implemented by distributions whose parameter values can be updated by a Markov increment
/// step from the current parameter value.
pub trait Markov
where
    Self : Distribution
{

    fn natural_mut<'a>(&'a mut self) -> DVectorSliceMut<'a, f64>;

    // If this distribution has a non-identity natural-to-canonical transformation,
    // return a mutable reference to the current state of the canonical transoformation.
    fn canonical_mut<'a>(&'a mut self) -> Option<DVectorSliceMut<'a, f64>>;

    // Updates current parameter state from a step trajectory
    /*fn update_from_step<'a>(&'a mut self, eta : &DRowSlice<'a, f64>) {
        self.natural_mut().tr_copy_from(&eta);
        //if let Some(canon) = self.canonical_mut {

        //}
    }*/

}*/

/// Retrieves a vector of fixed predictors and random observations from a distribution
pub(crate) fn retrieve_regression_data<L,O>(distr : &L) -> Option<(DMatrix<f64>, DVector<f64>)>
where
    L : Likelihood<O> + Distribution + ?Sized
{
    let y = distr.view_variable_values()?.column(0).clone_owned();
    let x = distr.view_fixed_values()?.clone_owned();
    Some((x, y))
}

/* A sufficient statistic compresses an iterator over observations O into a statically-known
vector of type [T; N].
pub struct Statistic<O, T, N> {
    sample : Option<Vec<O>>,
    func : Box<dyn Fn(impl Iterator<Item=&T>)->[T; N]
    value : Option<[T;N]>
}

pub trait Sample {

    fn statistic() -> Statistic<O, T, N>;
}

A un-conditional distribution needs to store only the statistic of dimensionality N.
A conditional distribution must store the full data set, to re-calculate the statistic
for every parameter value set at its parent. The statistic might or might not store this full data set
depending on the conditioning structure of the model.

A statistic compresses the n-dimensional distribution into a 1-dimensional distribution.
*/

/// Fixed is implemented by distributions whose partial conditionals have a analytic solution.
/// A multivariate normal, for example, can be held fixed at its non-random entries, leaving
/// the random entries as linear functions of the fixed values. Instead of being a distribution
/// over observations (like Likelihoods) or over parameters (like Priors), fixed distributions
/// have the role of storing the coefficients of the linear transformation required to produce
/// its conditional factor:
///
/// ```
/// let y = Normal::likelihood(&[1., 1.1, 2.1]);
/// let x = MultiNormal::fixed(&[&[1.,2.], &[3., 3.]], &[0.1, 0.2]);
/// y.condition(x);
/// ```
///
/// x will then have mean &[0.1, 0.2] and unit variance. Sampling from x generates coefficients, not observations.
/// Calcularing the log-probility of x happens with respect to the coefficients, not the observations.
/// But when we condition x on y, the mean of y will be x*b, not b.
///
/// Only MultiNormal implements fixed for now. This trait is the basis to implement maximum likelihood
/// (fixed implementor has covariance matrix zero) and bayesian regression models (fixed implementor
/// has non-zero covariance matrix).
///
/// The Fixed trait allow us to represent any model in the generalized linear family by a convergent graph
/// or random nodes. The parent variables are all held in a "fixed MultiNormal" where each element is the
/// (w * variate) expression; where w is a fixed weight and variate is a fixed realization. The marginal
/// correlations between those individual linear predictors and a converged-to dependent variable (y) define a full
/// generalized regression problem when we invert the precision matrix where the correlations are a single row (column).
pub trait Fixed<O>
where
    O : ?Sized
{

    fn fixed<'a>(values : impl Iterator<Item=&'a O>, coefs : impl Iterator<Item=&'a f64>) -> Self
    where
        O : 'a;

    fn observe_fixed<'a>(&mut self, values : impl Iterator<Item=&'a O>)
    where
        O : 'a;

    // fn observe_fixed<'a>(values : impl Iterator<Item=&'a Sample>, coefs : impl Iterator<Item=&'a f64>);
    // Updates an initially random distribution by fixig its values:
    // let m = MultiNormal::prior(&[0.2, 0.3]).fix(&[&[0.1, 0.2], &[1.2, 1.2]]);
    // fn fix(&mut self, obs : &[Sample]);
}

/// Latent is implement by distributions which could potentially hold n observations,
/// but do not. They have a single parameter n holding the expected sample size. This
/// value should be informed, because n == 1 means a prior distribution; n > 1 means
/// a latent variable in a factor graph. Moreover, when generating predictions, the distribution will
/// allocate n data points at its point in the factor graph. While Fixed is implemented by
/// multinormal to model conditional linear models, latent is implemented by normal and
/// multinormal to model latent linear models.
pub trait Latent {

    fn latent(n : usize) -> Self;

}

/// A mixture is defined by a linear combination of normal probability
/// distributions whose weights result from a categorical distribution draw,
/// which is a natural way to model a discrete
/// mutually-exclusive process affecting an independent continuous outcome:
/// p(y) = prod( p_i p(y_i)  )
/// This is essentially a marginalized distribution, where the p_i
/// are the discrete marginalization factors. The marginal is what is observed
/// at a random process (suppose we have k clusters, but we do not know
/// their centroids or dispersion); in this situation this representation essentially lets
/// us evaluate the log-probabilities of a proposed centroid vector and dispersion matrices
/// at all alternative outcomes, which are marginalied at the fixed values of the latent discrete
/// variable.
/// Since the inner probability does not factor with the rest of the log-
/// probabilities in the graph, the received outcome should be compared against
/// all possible combinations of the discrete outcome before being propagated
/// back into the graph.
/// This operation can be expressed as a product between a categorical
/// and a multivariate normal outcome: The dot-product between the categorical output
/// and the mean vector propagate a univariate normal distribution in the forward pass;
/// and the products of all potential realizations with the fixed values of the rhs define
/// a parameter vector to be propagated in the backward pass to both branches.
/// If we use this mixture and take the product again, the categorical can be interpreted
/// as being the LHS of the dot product with the row-stacked multivariate means, in which
/// case the mixture is selecting one of k possible multivariate realizations. By default, a
/// mixture will yield each element with the same probability. To change mixing probabilities, condition
/// the mixture on a Categorical distribution with the desired probabilities. Mixtures always have a
/// "default" element that the user has direct access, and a set of "remaining" elements owned by this
/// default element and accessed by mixture_factors and mixture_factors_mut. The probability vector
/// received by Categorical should contain only the remainig element probabilities; The probability
/// of the default element is 1.0 - sum(non_default).
pub trait Mixture
    where Self : Sized
{

    fn mix(self, other : Self) -> Self {
        Self::mixture([self, other])
    }

    fn mixture<const K : usize>(distrs : [Self; K]) -> Self;

    fn mixture_factors(&self) -> &[Box<Self>];

    fn mixture_factors_mut(&mut self) -> &mut Vec<Box<Self>>;

}

/*/// To move the natural parameter out of a distribution structure, so we can update its canonical
/// parameter vector by reading from it, the ideal solution would be:
/// let eta = mem::take(&mut self.eta);
/// (...compute at &mut self by reading from eta here...)
/// self.eta = eta;
/// But DVector<T> does not implement Default for this to work. We do the same computation here
/// manually, by appealing to the uninitialized constructor of Matrix. Our program will allow UB
/// if we access self.eta at its current state.
unsafe fn swap_with_uninitialized(eta : &mut DVector<f64>) -> DVector<f64> {
    let uninit = DVector::new_uninitialized(1).assume_init();
    mem::swap(eta, &mut uninit);
    uninit
}*/

/// Implemented by distributions that can represent time-ordered random processes.
/// Creating a distribution with D::stochastic(k) creates a distribution that uses
/// information from the k-past samples to determine its current state.
pub trait Stochastic {

    fn stochastic(order : usize, init : Option<&[f64]>) -> Self;

    fn evolve(&mut self, obs : &[f64]);

    fn state<'a>(&'a self) -> &'a [f64];

}

/*fn evolve_and_fit(distr : &mut D, alg : A, sample : &[f64])
where
    D : Stochastic + Estimator<A>
{
    distr.evolve(sample);
    distr.fit();
}*/

/*

// Implemented only for univariate distributions. May also be called Univariate.
// If Distribution does not have a scale parameter, scale(&self) always return 1.0.
pub trait Exponential {

    type Location;

    type Scale;

    fn location(&self) -> Self::Location;

    fn scale(&self) -> Option<Self::Scale>;

}

// Multivariate is a composition of univariate distributions.
pub trait Multivariate
where
    Self::Node : Univariate
{

    type Node = NormalNode;

    // Depth-first iteration over the network. Normal node is like a normal (has mean and variance)
    // but also has an association(.) method that returns associations with parent nodes.
    fn depth_iter(&'a self) -> impl Iterator<Item=NormalNode<'a>>;

    // Breadth-first iteration over the network
    fn breadth_iter(&'a self) -> impl Iterator<Item=NormalNode<'a>>;

}
*/


