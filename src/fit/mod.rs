use nalgebra::*;
use crate::prob::*;
use super::*;
use crate::sample::Sample;

/// Utilities to build inference algorithms.
pub mod utils;

/// Algorithm for approximating posteriors with multivariate normals
/// (Expectation Maximization; work in progress).
#[cfg(feature="gsl")]
pub mod optim;

/// Full posterior estimation via random walk simulation (Metropolis-Hastings algorithm)
/// and related non-parametric distribution representation (work in progress).
pub mod walk;

/// Linear estimation methods (Least squares and iteratively-reweighted least squares).
/// Those algorithms can be either treated as estimator in themselves
/// or be used as building block for more complex optimization or sampling strategies. 
pub mod linear;

// Online filtering algorithms
// pub mod filter

/// Trait shared by all inference algorithms, parametrized by the resulting posterior distribution. 
/// You might wish to implement special-purpose estimators that return an exact type of distribution
/// (for example, conjugate inference is implemented as Estimator<Beta>, Estimator<Gamma>, and so on)
/// or you might build a general purpose algorithm that return a non-parametric representation
/// (for example, Metropolis-Hastings is implemented as Estimator<Marginal>. Estimator does not care
/// how you instantiate your object. If you are building a general-purpose estimator, it is recommended
/// that your builder method is a generic function receiving Into<Model> as argument, which allows
/// your method to work both for methods built in the source code or specified as JSON files. You can then
/// match on the received model to decide the model admissibility for your algorithm.
pub trait Estimator<D>
    where
        // Self : ?Sized,
        D : Distribution //+ ?Sized + Posterior
{

    /// Runs the inference algorithm for the informed sample matrix,
    /// returning a reference to the modified model (from which
    /// the posterior information of interest can be retrieved).
    fn fit<'a>(&'a mut self, sample : &'a dyn Sample) -> Result<&'a D, &'static str>;
    
    /// If fit(.) has been called successfully at least once, returns the current state
    /// of the posterior distribution, whithout changing the algorithm state.
    fn posterior<'a>(&'a self) -> Option<&'a D>;
    
}

/*// Call the iteratively re-weighted least squares algorithm over random y (data).
// Assume x is already constant for the multinormal.
/// The formula for the IRLS solution is:
///
/// b_{t+1} = arg_min_b (X^T W_t X)^{-1} X^T W_t y [1]
/// Where W is calculated iteratively from the variance of the distribution predictions.
///
/// We interpret WX as the argument to scale_by to the parameter MultiNormal,
/// and require to use W y as our new observation at each iteration.
/// If b ~ MN(b,S) then WXb ~ MN(WXb, WX S (WX)^T); when we evaluate the
/// log-probability of linear op wrt Wy, we will have:
///
/// (Wy - WXb)^T (WX S (WX)^T)^-1 (Wy - WXb)^T [2]
/// which simplifies to [1]. The score becomes the difference
/// between the previous and current iteration.
pub struct IRLS<D>
where
    D : Distribution + Clone + Conditional<MultiNormal>
{
    distr : D
}

impl Estimator for IRLS<D>
where
    D : Distribution + Clone + Conditional<MultiNormal>
{

}*/

/*pub fn irls<D>(mut distr : D, y : DMatrix<f64>, x : DMatrix<f64>) -> Result<D, String>
where
    D : Distribution + Clone + Conditional<MultiNormal>
{
    println!("y = {}", y);
    println!("x = {}", x);
    assert!(y.nrows() == x.nrows());
    let mn_init : &MultiNormal = distr.view_factor().unwrap();
    let eta_init = mn_init.view_parameter(true).clone_owned();
    let mut weights = DMatrix::zeros(y.nrows(), y.nrows());
    weights.set_diagonal(&distr.var());

    let mn : &mut MultiNormal = distr.factor_mut().unwrap();
    //mn.scale_by(weights.clone() * &x);
    mn.scale_by(x.clone());

    let param = OptimParam::new()
        .init_state(eta_init)
        .preserve(100)
        .max_iter(100);

    // Here, we optimize over scaled_mu (x) but calcualte the gradient wrt the
    // eta vector.
    // Carry (y, X, W).
    let grad = |eta : &DVector<f64>, g : &mut (D, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)| -> DVector<f64> {

        /*let mut w = g.3.clone();
        update_weights(&g.0, &mut w);
        let mn_old : &MultiNormal = g.0.view_factor().unwrap();
        let old_lin_eta = mn_old.mean().clone();

        let mn : &mut MultiNormal = g.0.factor_mut().unwrap();
        mn.set_parameter((eta).into(), true);
        let lin_eta = mn.mean().clone_owned();

        let x = &g.2;
        let wx = w * x;
        mn.scale_by(wx.clone());
        let new_lin_eta = mn.mean().clone_owned();
        g.0.set_parameter((&new_lin_eta).into(), true);
        wx.transpose() * (new_lin_eta - old_lin_eta)*/

        let y = &g.1;
        let x = g.2.clone();
        // let w = g.3.clone();
        // update_weights(&g.0, &mut g.3);
        let y_pred = g.0.mean();
        let score = x.transpose() * /*&g.3 * */ (y_pred - y);
        println!("score = {}", score);
        score
    };
    let obj = |eta : &DVector<f64>, g : &mut (D, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)| -> f64 {
        // println!("param = {}", x);
        // During optimization, we will receive a vector of size one from the optimizer.
        // We must propagate to the actual natural parameter size (3).

        /*let mn : &MultiNormal = g.0.view_factor().unwrap();
        let y = &g.1;
        let x = &g.2;
        let w = &g.3;
        // let wx = w.clone() * x;
        let wy = (w.clone() * y).transpose();
        // mn.scale_by(wx.clone());
        //(-1.)*mn.log_prob(wy.slice((0, 0), (1, wy.ncols())), None)
        (-1.)*g.0.log_prob(((w.clone() * y)).slice((0, 0), (y.nrows(), 1)), None)*/

        let mut mn : &mut MultiNormal = g.0.factor_mut().unwrap();
        mn.set_parameter((eta).into(), true);
        let eta_lin = mn.mean().clone_owned();
        g.0.set_parameter((&eta_lin).into(), true);
        let y = &g.1;
        println!("eta = {}", eta);
        (-1.)*g.0.log_prob((y).slice((0, 0), (y.nrows(), 1)), None)
    };
    let mut optim = LBFGS::prepare(param, (distr, y, x, weights))
        .with_gradient(grad)
        .with_function(obj);
    let min = optim.minimize()
        .map_err(|e| format!("Minimization failed: {:?}", e) )?;
    println!("Minimum = {}", min);
    let mut distr = optim.take_data().0;
    let mn : &mut MultiNormal = distr.factor_mut().unwrap();
    mn.set_parameter((&min.value).into(), false);
    let mn_mean = mn.mean().clone_owned();
    distr.set_parameter((&mn_mean).into(), true);
    Ok(distr)
}

#[test]
fn logistic() {
    // Create random variable with logit value increasing as a function of X.
    let norm = Normal::new(100, None, None);
    let logit_noise = 0.1 * norm.sample();
    let mut bern = Bernoulli::new(100, None);
    let logit = DVector::from_fn(100, |i, _| -5.0 + i as f64 * 0.1 + logit_noise[i] );
    bern.set_parameter((&logit).into(), true);
    let mut mn = multinormal::MultiNormal::new_standard(3);
    let x = DMatrix::from_columns(&[
        DVector::from_element(100, 1.),
        2. * norm.sample().column(0).clone_owned(),
        3. * norm.sample().column(0).clone_owned()
    ]);
    let y = bern.sample();
    let bern = bern.condition(mn);
    println!("y = {}", y);
    println!("x = {}", x);
    println!("{:?}", irls(bern, y, x));
}*/

/*
/// Generic function used for testing that the mean of the posterior resulting from estimator e
/// converges to the bias vector informed by the user as the number of samples (collected by
/// resampling the user-informed sample) grows. Biasedness is formally defined by
/// E[t|theta] = theta asymptotically, where t is a statistic calculated from the data.
pub fn bias<P, R, C>(e : impl Estimator<P>, sample : &dyn Sample<R,C>, bias : &[f64]) {

}

/// Generic function used to test that the variance (or covariance diagonal) of the posterior
/// resulting from estimator e converges to the variance vector informed by the user as the number
/// of samples (collected by resampling the user-informed sample) grows.
pub fn variance<P, R, C>(e : impl Estimator<P>, sample : &dyn Sample<R, C>, variance : &[f64]) {

}

/// Generic function used to test if the mean of the posterior calculated by an an estimator E consistently approach
/// a "true" value, defiend by a generative process informed by the user as the last argument.
pub fn consistency<P, R, C>(e : impl Estimator<P>, sample : &dyn Sample<R, C>, true : impl Distribution) {

} 
*/


