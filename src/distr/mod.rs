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

pub trait Distribution
    where Self : Debug + Sized //+ Clone /*+ JointDistribution*/
{

    /// Returns the vector(s) that completely determine the output of log_prob and sample.
    /// For ExponentialFamily implementors, this is the canonical parameter (at the
    /// same scale of the distribution samples); For Unary and Binary operations
    /// this is the transformation function applied to both parameters; For non-parametric
    /// distributions such as histograms this is the full sample over which the histogram
    /// is calculated.
    fn mean<'a>(&'a self) -> &'a DVector<f64>;

    /// Set internal parameter vector at the informed value.
    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, natural : bool);

    fn mode(&self) -> DVector<f64>;

    fn var(&self) -> DVector<f64>;

    fn cov(&self) -> Option<DMatrix<f64>>;

    /// Log-probability evaluated at the vector y against self.parameter(). The column
    /// dimension of this sample is known at compile time; The row-dimension of the sample
    /// should match the number of sampled parameters at self.parameter(), and a panic
    /// occurs at a dimension mismatch. log_prob(.) reads parameter values from a
    /// vector owned by the distribution implementer. When evaluating the marginal
    /// posterior, it is useful to use this function to evaluate the fixed parameters;
    /// and evaluate the parameters varying at each iteration via cond_log_prob.
    fn log_prob(&self, y : DMatrixSlice<f64>) -> f64;

    /// Sample from the current distribution; If the sampling unit has multiple dimensions,
    /// they are represented over columns; If multiple units are sampled (if there are multiple
    /// entries for the parameter vector), a variable number of rows is emitted.
    fn sample(&self) -> DMatrix<f64>;

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
pub trait ConditionalDistribution<D>
    where
        Self : Distribution + Sized,
        D : Distribution
{

    /// Sets d as a factor for this distribution. After this method is called,
    /// sample(.) from self should be conditional on a sample(.) from d; and
    /// log_prob(.) of self should also depend on d.
    fn condition(self, d : D) -> Self;

    fn view_factor(&self) -> Option<&D>;

    fn take_factor(self) -> Option<D>;

    fn factor_mut(&mut self) -> Option<&mut D>;

}

// TODO load the sufficient stat as a field for conjugate.
#[derive(Debug, Clone)]
pub enum UnivariateFactor<D> {
    Empty,
    CondExpect(MultiNormal),
    Conjugate(D)
}

/// Generic trait shared by all exponential-family distributions.
/// The parameter vector of Univariate distributions represent a series of realizations
/// that are conditionally independent. But the parameter vector of a multivariate distribution
/// relates to a series of multivariate realizations of a single conditioning factor; to represent
/// conditional independence of multivariate relationships, use factor sharing. Shared sister factors
/// are hevily dependent, but conditioned on each of their paths, they are conditionally independent,
/// and that is why multivariate quantities are commonly represented via univariate distributions at
/// the nodes of the graph.
/// The choice of modelling univariate quantities with the same distribution implementor; but multivariate
/// quantities with different implementors has some justifications: First, all distributions can implement
/// indexing operations in a meaningul way (the user might want to inspect conditional expectations and
/// yield different histograms via normal[3] or normal[10] for a linear regression model for example, following
/// the same strategy of indexing multivariate quantities). Second, it is extremely common to assume homoscedasticity
/// but different linear conditional expected values, a situation that can be modelled as matrix-vector multiplication
/// easily. Assuming different nodes for each univariate realization would hurt performance because we could not use
/// matrix manipulations so directly; Third, there is a nice connection with the operation of accessing multivariate distribution
/// dimensions as matrix multiplication by a diagonal design matrix, which can just yield a univariate distribution with varying
/// conditional expectation: This connection suggests all distributions are multivariate; The univariate ones just have
/// diagonal scale factors defined exclusively by their location parameters. A series of univariate measurements can actually
/// be interpreted as a single realization of a multivariate variable; with location fixed at the estimated vector value
/// and scale completely determined by this location or another independent scale factor.
/// Multivariate quantities have to be implemented as separate nodes because the user
/// need to express situations where nodes share the same scale factor (as univariate distributions does)
/// and situations where the nodes have independent scale factors as different graphs.
/// The first sister node in a chain always keeps ownership of the parent factor, while new sister
/// nodes keep only a reference to the parent factor and keep ownership of the sister factor to the left.
/// Feeding data to any cond_log_prob(.) of a node with sisters require a matrix column dimensionality
/// that matches the number of sisters: If each child node of a group of size k has dimension d, the data
/// matrix should be k*d dimensional, where the last d columns are fed to the last sister, and so on.
/// To view a univariate distribution in its current state, use self.histogram().full() to return a vector
/// representation of its density. Multivariate distributions can be inspected by multiplying them with
/// constant design matrices that yield univariate quantities. For a diagonal choice of matrix, each
/// element will be a univariate distribution with a histogram representation:
/// let m = MultiNormal::new(5); let uv = x * m; for u in 0..5 { uv[i].histogram().full() }
pub trait ExponentialFamily<C>
    where
        C : Dim,
        Self : Distribution
{

    /// Univariate case:
    /// Returns size-1 vector if this distribution is conditioned on a conjugate
    /// or constant; Returns size-n vector if this distribution is conditioned on
    /// a size n sample.
    /// Multivariate (d) case:
    /// Returns 1 x d matrix if distribution is conditioned on a conjugate
    /// or constant; Returns n x d matrix if distribution is conditioned
    /// on a size n sample.
    fn link_inverse<S>(
        eta : &Matrix<f64, Dynamic, U1, S>
    ) -> Matrix<f64, Dynamic, U1, VecStorage<f64, Dynamic, U1>>
        where S : Storage<f64, Dynamic, U1>;

    /// Underlying transformation of the natural parameter, involving
    /// a division by a scale factor. This is a simple transformation of the location
    /// parameter for bounded distributions such as the Bernoulli; but involves a division
    /// by a separate parameter in unbounded distributions such as the normal. After this transformation,
    /// the parameter space becomes unbounded and can take any values over the real line, independent
    /// of the distribution.
    fn link<S>(
        theta : &Matrix<f64, Dynamic, U1, S>
    ) -> Matrix<f64, Dynamic, U1, VecStorage<f64, Dynamic, U1>>
        where S : Storage<f64, Dynamic, U1>;

    /// Transform the independent sample (with data arranged over rows)
    /// into a sufficient statistic matrix.
    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64>;

    /// Calculates the log-probability with respect to the informed sufficient
    /// statistic matrix.
    fn suf_log_prob(&self, y : DMatrixSlice<'_, f64>) -> f64;

    /// Normalization factor. Usually is a function of the
    /// dimensionality of the parameter, and is required for the distribution
    /// to integrate to unity. Function of a random sample. For univariate samples,
    /// this will be the element-specific base measure; For multivariate samples,
    /// that are evaluated against a single parameter; this will be a repeated value
    /// according to a given number of samples.
    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64>;

    fn update_log_partition<'a>(&'a mut self, eta : DVectorSlice<'_, f64>);

    fn update_grad(&mut self, eta : DVectorSlice<'_, f64>);

    fn grad(&self) -> &DVector<f64>;

    /// Function that captures the distribution invariances to
    /// location (first derivative) and scale (second derivative).
    /// This is a vector of the same size as the sample for univariate
    /// quantities, assuming the values according to the conditional expectation;
    /// but is a vector holding a single value for multivariate quantities, which are evaluated against a single parameter value.
    fn log_partition<'a>(&'a self) -> &'a DVector<f64>;

    fn prob(&self, y : DMatrixSlice<f64>) -> f64 {
        // (Assume self does not have any factor,
        // since log_prob will evaluate whole graph)
        let bm = Self::base_measure(y.clone());
        let mut unn_p = DVector::zeros(y.nrows());
        for (i, _) in y.row_iter().enumerate() {
            unn_p[i] = self.log_prob(y.rows(i,1)).exp();
        }
        let p = bm.component_mul(&unn_p);
        let joint_p = p.iter().fold(1., |jp, p| jp * p);
        joint_p
    }


}

/// Inference algorithm, parametrized by the distribution output, which can be either
/// a full posterior or an approximation.
pub trait Estimator<D>
    where
        Self : Sized,
        D : Distribution
{

    /// Return a reference to the posterior distribution.
    /// If the algorithm will not be called anymore, just
    /// move the posterior to the current environment:
    /// let post = *(model.posterior().unwrap());
    fn fit<'a>(&'a mut self, y : DMatrix<f64>) -> Result<&'a D, &'static str>;

}

trait ConditionalLikelihood
    where Self : ExponentialFamily<U1>
{

    /// Calculates the univariate log-probability, passing
    /// the log-probability buffer forward. The univariate
    /// log-probability is based on the scalar multiplication of
    /// each natural parameter realization with each sample at y.
    fn cond_log_prob(&self,
        eta_cond : DVectorSlice<'_, f64>,
        y : DMatrixSlice<'_, f64>
    ) -> f64 {
        assert!(y.ncols() == 1);
        let mut lp = 0.0;
        for ((e, y), l) in eta_cond.iter().zip(y.column(0).iter()).zip(self.log_partition()) {
            lp += e * y - l
        };
        lp
    }
}




