use nalgebra::*;
use nalgebra::storage::*;
use std::fmt::Debug;
use std::ops::AddAssign;
use crate::decision::BayesFactor;
use std::fmt::Display;
use crate::sim::RandomWalk;
use anyhow;
use thiserror::Error;

pub mod poisson;

pub use poisson::*;

pub mod beta;

pub use beta::*;

pub mod bernoulli;

pub use bernoulli::*;

pub mod gamma;

pub use gamma::*;

pub mod normal;

pub use normal::*;

pub mod multinormal;

pub use multinormal::*;

pub mod wishart;

pub use wishart::*;

pub mod categorical;

pub use categorical::*;

pub mod dirichlet;

pub use dirichlet::*;

pub mod vonmises;

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
where Self : Debug + Display //+ Sized
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
/// and posterior factors after inference. Undirected probabilistic relationships are implemented
/// within dedicated structures (see MarkovChain and MarkovField).
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

    /// Returns a mutable reference to the parent factor.
    fn factor_mut(&mut self) -> Option<&mut D>;

}

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
    Empty,
    CondExpect(MultiNormal),

    // TODO load the sufficient stat as a field for conjugate.
    Conjugate(D)
}

fn univariate_log_prob<D>(
    y : DMatrixSlice<f64>,
    x : Option<DMatrixSlice<f64>>,
    factor : &UnivariateFactor<D>,
    eta : &DVector<f64>,
    lp : &DVector<f64>, /*f64*/
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
    (eta_s.component_mul(&y.slice((0, 0), (y.nrows(), 1))) - lp).sum() + factor_lp
}

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
    /// but is a vector holding a single value for multivariate quantities, which are evaluated against a single parameter value.
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

/// Inference algorithm, parametrized by the distribution output.
pub trait Estimator<D>
    where
        Self : Sized,
        D : Distribution
{

    /// Runs the inference algorithm for the informed sample matrix,
    /// returning a reference to the modified model (from which
    /// the posterior information of interest can be retrieved).
    fn fit<'a>(&'a mut self, y : DMatrix<f64>, x : Option<DMatrix<f64>>) -> Result<&'a D, &'static str>;

}

/// Implemented by distributions which can have their
/// log-probability evaluated with respect to a random sample directly.
/// Implemented by Normal, Poisson and Bernoulli. Other distributions
/// require that their log-probability be evaluated by using suf_log_prob(.)
/// by passing the sufficient statistic calculated from the sample.
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
pub trait Likelihood<C>
    where
        Self : ExponentialFamily<C> + Distribution + Sized,
        C : Dim
{

    /// General-purpose comparison of two fitted estimates, used for
    /// determining predictive accuracy, running cross-validation, etc.
    /// Comparisons can be made between two fitted models
    /// for purposes of hyperparameter tuning or model selection; between
    /// a fitted model and a saturated model for decision analysis; or between
    /// a fitted model and a null model for hypothesis testing.
    fn compare<'a, D>(&'a self, other : &'a D) -> BayesFactor<'a, Self, D>
        where D : Distribution + Sized
    {
        BayesFactor::new(&self, &other)
    }

    /*/// Here, we approximate the relative entropy, or KL-divergence
    /// E[log_p(x) - log_q(x)] by the average of a few data point pairs (y, x)
    fn entropy<'a, D>(&'a self, other : &'a D, y : DMatrixSlice<'_, f64>, x : Option<DMatrixSlice<'_, f64>>) -> f64
        where D : Distribution + Sized
    {
        BayesFactor::new(&self, &other).log_diff(y, x)
    }*/


    /// Returns the distribution with the parameters set to its
    /// gaussian approximation (mean and standard error).
    fn mle(y : DMatrixSlice<'_, f64>) -> Result<Self, anyhow::Error>;
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

    fn visit_factors<F>(&mut self, f : F) where F : Fn(&mut dyn Posterior);

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

    /// The conditional log-probability evaluation works because
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
    }

    // Iterate over sister nodes if Factor; or returns a single distribution if
    // not a factor.
    // pub fn iter_sisters() -
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
/// methods, instead of returning None, will return Some(MultiNormal) and/or Some(RandomWalk)
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

    fn trajectory(&self) -> Option<&RandomWalk>;

    fn trajectory_mut(&mut self) -> Option<&mut RandomWalk>;

    // Mark this variable fixed (e.g. at its current MLE) to avoid using it further as part of the
    // Inference algorithms by querying it via the fixed() method.
    // fn fix(&mut self)

    // Verify if this variable has been fixed by calling self.fix() at a previous iteration.
    // fn fixed(&self)

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

/*
/// A MarkovChain is a directed cyclic graph of categorical distributions.
/// It is the discrete analog of the RandomWalk structure.
/// Categoricals encode state transition probabilities (which inherit all the
/// properties of categoricals, such as conditioning on MultiNormal factors,
/// useful to model the influence of external factors at the transition probabilities).
///
/// Transition probabilities are required only to be conditionally independent,
/// but they might be affected by factor-specific external variables.
struct MarkovChain {

    /// A state is simply a categorical holding transition probabilities.
    /// Since categoricals can be made a function of a multinormal (multinomial regression),
    /// the transition probabilities can be modelled as functions of external features.
    states : Vec<Categorical>,

    /// The target state to which each transition refers to is the entry at the dst
    /// vector. Each entry at the inner vector is an index of the states vector
    /// (including the current one). Transition targets are not required to be of
    /// the same size, and empty inner vectors mean final states. Transitions might
    /// refer to any categorical in the states vector, including the current state.
    dst : Vec<Vec<usize>>,

    /// The limit field determines the maximum transition size. Without this field,
    /// recursive chains would repeat forever.
    limit : usize,

    curr_state : usize
}

pub enum Transition {

    /// Explore all transition possibilities
    Any,

    /// Only transition to the highest probability
    Highest,

    /// Accept only probabilities that have minimum value
    Minimum(f64),

    /// Accept only the n-best probabilities for any given transition
    Best(usize)

}

impl MarkovChain {

    /// Return an exhaustive list of all possible trajectories and
    /// their respective joint probabilities, ordered from the most likely trajectory to
    /// the least likely. Using trajectories.first() yields the MAP estimate for the markov process.
    /// Trajectores start at the informed state and end until either a final node is found
    /// or the state transition limit is reached.
    fn trajectories(&self, from : usize, rule : Transition) -> Vec<(Vec<usize>, f64)>;

    /// Generate a random trajectory
    fn sample(&self, n : usize, rule : Transition) -> Vec<<Vec<usize>>;

}

/// Use the curr_state method to walk into some state. Might yield mutable references so
/// the categoricals may be updated with external data.
impl Iterator for MarkovChain {

}

impl Extend for MarkovChain {

    /// Receives an iterator over the tuple (Categorical, Vec<usize>)
    fn extend<T>(&mut self, iter: T) {

    }
}

/// HiddenMarkov wraps a Markov chain for which only the realizations
/// of corresponding continuous distributions are seen (the observed variables
/// are mixture distributions). A index realization
/// i means that the observation is a draw by indexing the obs vector at
/// index i. Using observed continuous states conditional on discrete states
/// naturally accomodate translation/scale variants expected in sound/image
/// recognition problems.
struct HiddenMarkov {
    chain : MarkovChain
    obs : Vec<MultiNormal>
}

*/


