use nalgebra::*;
use nalgebra::storage::*;
use std::fmt::Debug;
use std::ops::AddAssign;
use crate::decision::BayesFactor;

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
pub trait Distribution
    where Self : Debug + Sized
{

    /// Returns the expected value of the distribution, which is a function of
    /// the current parameter vector.
    fn mean<'a>(&'a self) -> &'a DVector<f64>;

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

    /// Evaluates the log-probability of the sample y with respect to the current
    /// parameter state. This method just dispatches to a sufficient statistic log-probability
    /// or to a conditional log-probability evaluation depending on the conditioning factors
    /// of the implementor. The samples at matrix y are assumed to be independent over rows (or at least
    /// conditionally-independent given the current factor graph). Univariate factors require that
    /// y has a single column; Multivariate factors require y has a number of columns equal to the
    /// distribution parameter vector.
    fn log_prob(&self, y : DMatrixSlice<f64>) -> f64;

    /// Sample from the current distribution; If the sampling unit has multiple dimensions,
    /// they are represented over columns; If multiple units are sampled (if there are multiple
    /// entries for the parameter vector), a variable number of rows is emitted. Samples should
    /// follow the same structure as the argument to log_prob(.).
    fn sample(&self) -> DMatrix<f64>;

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

// TODO load the sufficient stat as a field for conjugate.
#[derive(Debug, Clone)]
pub enum UnivariateFactor<D> {
    Empty,
    CondExpect(MultiNormal),
    Conjugate(D)
}

/// Generic trait shared by all exponential-family distributions. Encapsulate
/// all expressions necessary to build a log-probability with respect to a
/// sufficient statistic.
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

    /// The unnormalized log-probability is always defined as the inner product of a sufficient
    /// statistic with the natural parameter minus a term dependent on the parameter
    /// alone. This method updates this term for every parameter update.
    fn update_log_partition<'a>(&'a mut self, eta : DVectorSlice<'_, f64>);

    /// The gradient of an exponential-family distribution is a linear function
    /// of the sufficient statistic (constant) and the currently set natural parameter.
    fn update_grad(&mut self, eta : DVectorSlice<'_, f64>);

    /// Retrieves the gradient of self, with respect to the currently set natural
    /// parameter and sufficient statistic.
    fn grad(&self) -> &DVector<f64>;

    /// Function that captures the distribution invariances to
    /// location (first derivative) and scale (second derivative).
    /// This is a vector of the same size as the sample for univariate
    /// quantities, assuming the values according to the conditional expectation;
    /// but is a vector holding a single value for multivariate quantities, which are evaluated against a single parameter value.
    fn log_partition<'a>(&'a self) -> &'a DVector<f64>;

    /// Normalized probability of the independent sample y.
    fn prob(&self, y : DMatrixSlice<f64>) -> f64 {
        // (Assume self does not have any factor,
        // since log_prob will evaluate whole graph)
        let bm = Self::base_measure(y.clone());
        let mut unn_p = DVector::zeros(y.nrows());
        for (i, _) in y.row_iter().enumerate() {
            unn_p[i] = self.log_prob(y.rows(i,1)).exp();
            println!("lp = {}", unn_p[i]);
        }
        let p = bm.component_mul(&unn_p);
        let joint_p = p.iter().fold(1., |jp, p| jp * p);
        joint_p
    }

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
    fn fit<'a>(&'a mut self, y : DMatrix<f64>) -> Result<&'a D, &'static str>;

}

/// Mutable iterator over a probabilistic graph.
/// Starts at the immediate factors of a likelihood node
/// and iterate up to all elements without any factors.
/// Location factors are visited before scale factors,
/// and every path is explored in a depth-first fashion.
pub struct Factors<'a> {
    factors : Vec<&'a mut dyn Posterior>
}

impl<'a> Factors<'a> {

    pub fn new_empty() -> Self {
        Self{ factors : Vec::new() }
    }

    pub fn push(&'a mut self, p : &'a mut dyn Posterior) {
        self.factors.push(p);
    }

    pub fn aggregate(mut self, p : &'a mut dyn Posterior) -> Self {
        self.factors.push(p);
        self
    }

    /// The visit(.) method takes a generic closure and an optional sequence
    /// of generic data references that can carry some data to apply
    /// to each node in the graph.
    pub fn visit<F, D>(&mut self, f : F, opt_data : Option<&[D]>) -> Result<(), &'static str>
        where F : Fn(&mut dyn Posterior, Option<&D>)
    {
        let mut ix = 0;
        for factor in self.factors.iter_mut() {
            if let Some(data) = opt_data {
                if let Some(d) = data.get(ix) {
                    f(*factor, Some(d));
                    ix += 1;
                } else {
                    return Err("Not enought data entries");
                }
            } else {
                f(*factor, None);
            }
        }
        match opt_data {
            Some(data) => if data.len() == ix {
                Ok(())
            } else {
                Err("Excess of data entries")
            },
            None => Ok(())
        }
    }

}

impl<'a> Iterator for Factors<'a> {

    type Item = &'a mut dyn Posterior;

    fn next(&mut self) -> Option<&'a mut dyn Posterior> {
        if self.factors.len() == 0 {
            None
        } else {
            Some(self.factors.remove(0))
        }
    }

}

impl<'a> From<Factors<'a>> for Vec<&'a mut dyn Posterior> {

    fn from(f : Factors<'a>) -> Vec<&'a mut dyn Posterior> {
        f.factors
    }
}

/// Implemented by distributions which can have their
/// log-probability evaluated with respect to a random sample directly.
/// Implemented by Normal, Poisson and Bernoulli. Other distributions
/// require that their log-probability be evaluated by using suf_log_prob(.)
/// by passing the sufficient statistic calculated from the sample.
pub trait Likelihood<C>
    where
        Self : ExponentialFamily<C> + Distribution + Posterior,
        C : Dim
{

    /// General-purpose comparison of two fitted estimates, used for
    /// determining predictive accuracy, running cross-validation, etc.
    /// Comparisons can be made between two fitted models
    /// for purposes of hyperparameter tuning or model selection; between
    /// a fitted model and a saturated model for decision analysis; or between
    /// a fitted model and a null model for hypothesis testing.
    fn compare<'a, D>(&'a self, other : &'a D) -> BayesFactor<'a, Self, D>
        where D : Distribution
    {
        BayesFactor::new(&self, &other)
    }

    /// Returns the mean maximum likelihood estimate and its standard error.
    fn mle(y : DMatrixSlice<'_, f64>) -> (f64, f64) {
        (Self::mean_mle(y), Self::se_mle(y))
    }

    /// Returns a mean estimate using maximum likelihood estimation.
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
    }

    fn factors_mut(&mut self) -> Factors {
        let (fmut_a, fmut_b) = self.dyn_factors_mut();
        let factors_a = if let Some(a) = fmut_a {
            a.aggregate_factors(Factors::new_empty())
        } else {
            Factors::new_empty()
        };
        let factors_b = if let Some(b) = fmut_b {
            b.aggregate_factors(factors_a)
        } else {
            factors_a
        };
        factors_b
    }

    /// The conditional log-probability evaluation works because
    /// the sufficient statistic of Likelihood implementors is
    /// just the sum of the outcomes. Under this situation, the
    /// log-probability of the sufficient stat should equal the
    /// log-probability of the component-wise multiplication of
    /// the natural parameter with the individual samples.
    fn cond_log_prob(&self, y : DMatrixSlice<'_, f64>) -> f64 {
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
    }
}

pub trait Posterior {

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

    fn aggregate_factors<'a>(&'a mut self, factors : Factors<'a>) -> Factors<'a> {
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
    }

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Posterior>, Option<&mut dyn Posterior>);

    // fn approximate(&self) -> Option<&D>;
}


