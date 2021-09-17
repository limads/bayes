use std::fmt::Debug;
use nalgebra::*;
use crate::prob::*;
use std::fmt::{self, Display, Formatter};
use std::error::Error;

/// Optimization routines, via bindings to GNU GSL.
mod gsl;

// Expectation-maximization algorithm (Work in progress).
// pub mod em;

// pub use em::*;

mod approx;

#[derive(Debug, Clone)]
pub struct OptimParam {

    /// Maximum number of parameter iterations to store.
    iter_memory : usize,

    /// Maximum number of iterations the algorithm will run.
    max_iter : usize,

    init : DVector<f64>

    //obj : Option<Box<FnMut(&DVector<f64>, &mut T)>>
}

impl OptimParam {

    pub fn new() -> Self {
        Self{ init : DVector::zeros(1), max_iter : 100, iter_memory : 100 }
    }

    pub fn max_iter(mut self, iter : usize) -> Self {
        self.max_iter = iter;
        self
    }

    pub fn preserve(mut self, mem : usize) -> Self {
        self.iter_memory = mem;
        self
    }

    pub fn init_state(mut self, state : DVector<f64>) -> Self {
        self.init = state;
        self
    }
}

#[derive(Clone, Debug)]
pub enum MinError{

    /// Optimizer achieved maximum number of iterations before finding a global minima.
    MaxIter(usize),

    /// Optimizer could not be initialized due to some error.
    Init(String),

    /// Generic execution error.
    Exec(String)
}

impl Display for MinError {

    fn fmt(&self, f : &mut Formatter<'_>) -> Result<(), fmt::Error> {
        let msg = match self {
            MinError::MaxIter(n) => format!("Maximum number of iterations reached ({})", n),
            MinError::Init(msg) => format!("Could not initialize optimizer: {}", msg),
            MinError::Exec(msg) => format!("Error running optimizer: {}", msg)
        };
        write!(f, "Optimizer error: {}", msg)
    }

}

impl Error for MinError{ }

/// Result of an optimization algorithm (success or failure).
#[derive(Clone, Debug)]
pub struct Minimum {

    /// Last function domain nstate.
    pub value : DVector<f64>,

    /// Last function evaluated value.
    pub eval : f64,

    pub iter : usize
}

impl Display for Minimum {

    fn fmt(&self, f : &mut Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Minimum found after {} iterations (x = {};\nf(x) = {})",
            self.iter, self.value, self.eval
        )
    }

}

/// Generic API implemented by optimization algorithms.
/// Optimizers might keep a history of their iterations, which
/// can be retrieved by the value_trajectory and eval_trajectory
/// methods. Optimizers should be initialized by the prepare method,
/// which might take arbitrary data, which the user has access
/// via a mutable reference from the supplied closure (if no additional
/// data is required, pass unit to prepare).
pub trait Optimizer<T : Clone + Sized> {

    /// Start the optimizer with the given parameters.
    fn prepare(
        param : OptimParam,
        data : T
    ) -> Self;

    /// Returns parameter values across optimizer iterations.
    /// The result should be a "tall" matrix with states written
    /// over the rows. Returned values should be the minimum of
    /// the number of iterations or the parameter iter_memory.
    fn value_trajectory(&self) -> &DMatrix<f64>;

    /// Returns objective evaluations across optimizer iterations,
    /// stored at a single column vector.
    fn eval_trajectory(&self) -> &DVector<f64>;

    /// Returns the first derivative evaluations, following the
    /// same convention of value_trajectory, if applicable
    /// for the implementor.
    fn grad_trajectory(&self) -> Option<&DMatrix<f64>>;

    /// Returns a constant second derivative approximation
    /// based on the trajectory of first derivative evaluation
    /// (if available) or on the (backward) finite differences
    /// calculated from the optimizer trajectory.
    fn hessian_approx(&self) -> DMatrix<f64> {
        match self.grad_trajectory() {

            /// Approximate from gradient trajectory
            Some(grad_traj) => {
                approx::hessian(grad_traj.clone())
            },

            /// Approximate from value trajectory
            None => {
                let xs = self.value_trajectory().clone();
                let ys = self.eval_trajectory();
                let dxs = approx::gradient(xs, ys);
                approx::hessian(dxs)
            }
        }
    }

    /// Supplies the objective to be minimized via a closure. If no
    /// closure is supplied here before minimization, the process will
    /// fail.
    fn with_function<F>(self, f : F) -> Self
    where
        F : FnMut(&DVector<f64>, &mut T)->f64 + 'static;

    /// Run the minimization using the supplied objective and (if applicable)
    /// the gradient.
    fn minimize(&mut self) -> Result<Minimum, MinError>;

}

/*/// Generic n-dimensional approximation to another distribution. Built
/// by finding the Gaussian that best approximate the target distribution
/// via the finite-difference method applied to an optimizer trajectory.
pub struct Approximation {
    approx : MultiNormal,
    optim : LBFGS<MultiNormal>
}*/

/*impl Approximation {

    /// Given the univariate (1 row) or multivariate (p rows)
    /// convex optimization trajectory at the wide eta matrix (realizations over rows),
    /// and the corresponding log-probability realizations of the objective
    /// at the second argument, build a multivariate normal approximation
    /// to the distribution from which the log-probability comes from.
    fn build(mut eta : DMatrix<f64>, lp_traj : DVector<f64>) -> Self {
        assert!(eta.ncols() == lp_traj.nrows());
        let mu_approx = eta.column(eta.ncols() - 1).clone_owned();
        let prec_approx = unimplemented!();

        // Obtain approximate covariance from approximate hessian via QR inversion
        let sigma_approx = Self::invert_scale(&prec_approx);
        MultiNormal::new(mu_approx, sigma_approx).unwrap();
    }

    fn update(&mut self, eta : DMatrix<f64>) {

    }
}*/

// Perhaps leave T as plain ref here and allow mutation only at Objective?
type Gradient<T> = Box<dyn FnMut(&DVector<f64>, &mut T)->DVector<f64>>;

// Perhaps use move semantics? FnMut(mut DVector<f64>, &mut T)->DVector<f64>>.
// The user should be able to still use a move || closure here, to send data
// he has no interest in recovering?
type Objective<T> = Box<dyn FnMut(&DVector<f64>, &mut T)->f64>;

/// Wraps [GSL's](https://www.gnu.org/software/gsl/doc/html/multimin.html) implementation
/// of the Limited-memory Broyden-Fletcher-Goldfarb-Shanno quasi-newton optimization algorithm
/// using the Optimizer trait.
///
/// # Example
///
/// ```
/// use bayes::optim::*;
/// use nalgebra::*;
///
/// let param = OptimParam::new()
///    .init_state(DVector::from_element(1, 1.))
///    .preserve(100)
///    .max_iter(100);
/// let grad = |x : &DVector<f64>, t : &mut ()| -> DVector<f64> {
///     DVector::from_element(1, 6. * x[0])
/// };
/// let obj = |x : &DVector<f64>, t : &mut ()| -> f64 {
///     3.*x[0].powf(2.) + 5.
/// };
/// let mut optim = LBFGS::prepare(param, ())
///     .with_gradient(grad)
///     .with_function(obj);
/// optim.minimize().map(|min| println!("{}", min) )
///    .expect("Minimization failed");
/// ```
pub struct LBFGS<T : Sized + Clone> {

    param : OptimParam,

    value : DMatrix<f64>,

    eval : DVector<f64>,

    grad_hist : DMatrix<f64>,

    data : T,

    grad : Option<Gradient<T>>,

    obj : Option<Objective<T>>

}

impl<T : Sized + Clone> LBFGS<T> {

    /// Informs the gradient. If no function is passed here, the optimizer
    /// will fail at minimization.
    pub fn with_gradient<G>(mut self, g : G) -> Self
    where
        G : FnMut(&DVector<f64>, &mut T)->DVector<f64> + 'static
    {
        self.grad = Some(Box::new(g));
        self
    }

    /// Retrieves the user-supplied data at prepare(.) by consuming self.
    pub fn take_data(mut self) -> T {
        self.data
    }

    /// Retrieves the user-supplied data at prepare(.) via cloning.
    pub fn clone_data(&self) -> T {
        self.data.clone()
    }
}

impl<T : Sized + Clone> Optimizer<T> for LBFGS<T> {

    /// Start the optimizer with the given parameters.
    fn prepare(param : OptimParam, data : T) -> Self {
        let eval = DVector::zeros(param.iter_memory);
        let value = DMatrix::zeros(param.init.nrows(), param.iter_memory);
        let grad_hist = value.clone();
        Self { param, value, eval, grad : None, grad_hist, data, obj : None }
    }

    /// Returns parameter values across optimizer iterations.
    /// The result should be a "tall" matrix with states written
    /// over the rows. Returned values should be the minimum of
    /// the number of iterations or the parameter iter_memory.
    fn value_trajectory(&self) -> &DMatrix<f64> {
        &self.value
    }

    /// Returns objective evaluations across optimizer iterations,
    /// stored at a single column vector.
    fn eval_trajectory(&self) -> &DVector<f64> {
        &self.eval
    }

    fn with_function<F>(mut self, f : F) -> Self
    where
        F : FnMut(&DVector<f64>, &mut T)->f64 + 'static
    {
        self.obj = Some(Box::new(f));
        self
    }

    /// F is a function or closure which takes a reference to a state vector and possibly
    /// mutate a state T across iterations. T might not be needed (in which case unit might be
    /// passed), or might hold information such as the function gradient.
    fn minimize(&mut self) -> Result<Minimum, MinError> {
        use gsl::bfgs::minimize_with_grad;

        let bx_g = self.grad.take()
            .ok_or(MinError::Init(String::from("No gradient informed")))?;
        let bx_f = self.obj.take()
            .ok_or(MinError::Init(String::from("No objective informed")))?;

        let ans = minimize_with_grad(
            self.param.init.clone(),
            self.data.clone(),
            bx_f,
            bx_g,
            Some(&mut self.value),
            Some(&mut self.grad_hist),
            Some(&mut self.eval),
            self.param.max_iter
        );
        match ans {
            Ok((value, _, eval, iter, del_t)) => {
                let n_remove = self.value.ncols() - iter;
                self.data = del_t;
                self.value = self.value.clone()
                    .remove_columns(iter, n_remove);
                self.grad_hist = self.grad_hist.clone()
                    .remove_columns(iter, n_remove);
                self.eval = self.eval.clone()
                    .remove_rows(iter, n_remove);
                Ok(Minimum{ value, eval, iter })
            },
            Err(s) => Err(MinError::Exec(s))
        }
    }

    fn grad_trajectory(&self) -> Option<&DMatrix<f64>> {
        Some(&self.grad_hist)
    }

}

/*#[test]
fn optim_distr() {

    use crate::distr::{Distribution, ExponentialFamily, Bernoulli};
    use nalgebra::*;

    let b = Bernoulli::new(1, Some(0.5));
    let y = DMatrix::from_column_slice(5, 1, &[1., 1., 0.]);

    let param = OptimParam::new()
        .init_state(DVector::from_element(1, 1.))
        .preserve(100)
        .max_iter(100);
    let grad = |x : &DVector<f64>, t : &mut ()| -> DVector<f64> {
        let s = b.grad(y.slice((0, 0), (3, 1)), None);
        s
    };
    let obj = |x : &DVector<f64>, t : &mut ()| -> f64 {
        b.log_prob(y.slice((0, 0), (3, 1)), None)
    };
    let mut optim = LBFGS::prepare(param, ())
        .with_gradient(grad)
        .with_function(obj);
    optim.minimize().map(|min| println!("{}", min) )
        .expect("Minimization failed");
}*/



