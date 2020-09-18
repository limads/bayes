use nalgebra::*;
use crate::distr::*;
use super::*;
use crate::distr::multinormal;

/// A hessian is approximated from two first derivative steps as:
/// (1/(f(t) - f(t-1)) * [x][x]^T = [Δx][Δx]^T (unscaled outer product of the
/// domain vectors, which equals the simple outer product of gradient
/// vectors). By receiving the gradient (dxs) the unscaling step
/// is already assumed to have taken place. Since the
/// steps/gradients are arranged as  column vectors, we can do the
/// outer product and averaging step at the same time. For convex functions
/// the hessian is a constant capturing the n-dimensional dispersion, and
/// there is no loss of information in using it instead of the optimizer
/// gradient trajectory to describe a n-dimensional surface.
pub fn hessian(mut dxs : DMatrix<f64>) -> DMatrix<f64> {

    // Build numerical hessian approximation: delta^T * delta
    // (actually this expression transposed since we are working
    // with a tall domain matrix).
    let hess_approx = dxs.clone() * dxs.transpose();

    // Guarantees symmetry and positive-definiteness
    let hess = multinormal::approx_pd(hess_approx);

    hess
}

/// The finite difference method transform a trajectory of function
/// evaluations ys = f(xs) into a trajectory of first derivative
/// approximations. For convex functions, the first derivative is
/// a linear function of the parameter vector.
pub fn gradient(mut xs : DMatrix<f64>, ys : &DVector<f64>) -> DMatrix<f64> {

    assert!(xs.ncols() >= 2);
    assert!(ys.nrows() >= 2);

    let xs_lag = xs.clone();

    // build the numerical gradient approximation using
    // backward difference: delta_i = lp_{i} / (x_{i} - x_{i-1})
    let delta_iter = xs.column_iter_mut()
        .skip(1)
        .zip(xs_lag.column_iter())
        .enumerate();
    for (i, (mut x_curr, x_last)) in delta_iter {
        x_curr -= x_last;
        x_curr.unscale_mut(&ys[i+1] - &ys[i]);
    }

    let delta = xs.remove_column(0);
    delta
}

#[test]
fn approx_gamma() {

    use crate::distr::*;
    use super::*;

    let g = Gamma::new(0.5, 0.5);
    let param = OptimParam::new()
        .init_state(DVector::from_column_slice(&[10.0, 10.0]))
        .preserve(100)
        .max_iter(100);
    let grad = |dx : &DVector<f64>, g : &mut (Gamma, DMatrix<f64>)| -> DVector<f64> {
        let grad = g.0.grad((&g.1).into(), None);
        println!("grad = {}", grad);
        grad
    };
    let obj = |x : &DVector<f64>, g : &mut (Gamma, DMatrix<f64>)| -> f64 {
        println!("param = {}", x);
        g.0.set_parameter((x).into(), true);
        let min = (-1.)*g.0.suf_log_prob((&g.1).into());
        println!("min = {}", min);
        min
    };
    let val = DMatrix::from_column_slice(2, 1, &[1., 1.]);
    let mut optim = LBFGS::prepare(param, (g, val))
        .with_gradient(grad)
        .with_function(obj);
    optim.minimize().map(|min| println!("{}", min) )
        .expect("Minimization failed");
}

/// Finds the MLE via optimization for an unscaled distribution such as the Poisson or Bernoulli
fn optimize_mle<D, C>(distr : D, data : DMatrix<f64>) -> Result<D, String>
where
    C : Dim,
    D : Distribution + ExponentialFamily<C>,
    (D, DMatrix<f64>) : Clone
{
    // Establish the 1D parameter vector.
    let param = OptimParam::new()
        .init_state(distr.view_parameter(true).rows(0,1).clone_owned())
        .preserve(100)
        .max_iter(100);
    let grad = |x : &DVector<f64>, g : &mut (D, DMatrix<f64>)| -> DVector<f64> {
        let x = if x.nrows() == 1 {
            DVector::from_element(g.1.nrows(), x[0])
        } else {
            x.clone_owned()
        };
        g.0.set_parameter((&x).into(), true);
        let grad = (-1.)*g.0.grad((&g.1).into(), None);
        println!("grad = {}", grad);
        grad
    };
    let obj = |x : &DVector<f64>, g : &mut (D, DMatrix<f64>)| -> f64 {
        println!("param = {}", x);

        // During optimization, we will receive a vector of size one from the optimizer.
        // We must propagate to the actual natural parameter size (3).
        let x = if x.nrows() == 1 {
            DVector::from_element(g.1.nrows(), x[0])
        } else {
            x.clone_owned()
        };
        g.0.set_parameter((&x).into(), true);
        let min = (-1.) * g.0.log_prob((&g.1).into(), None);
        println!("min = {}", min);
        min
    };
    let mut optim = LBFGS::prepare(param, (distr, data))
        .with_gradient(grad)
        .with_function(obj);
    let min = optim.minimize()
        .map_err(|e| format!("Minimization failed: {:?}", e) )?;
    println!("Minimum = {}", min);
    let mut distr = optim.take_data().0;
    distr.set_parameter((&min.value).into(), true);
    Ok(distr)
}

/// Finds the MLE via optimization of a scaled distribution (the normal),
/// conditional on the MLE of its scalar factor for each natural parameter iteration.
fn optimize_mle_scaled(distr : Normal, data : DMatrix<f64>) -> Result<Normal, String>
{
    let param = OptimParam::new()
        .init_state(distr.view_parameter(true).rows(0,1).clone_owned())
        .preserve(100)
        .max_iter(100);
    let grad = |x : &DVector<f64>, g : &mut (Normal, DMatrix<f64>)| -> DVector<f64> {
        let x = if x.nrows() == 1 {
            DVector::from_element(g.1.nrows(), x[0])
        } else {
            x.clone_owned()
        };
        g.0.set_parameter((&x).into(), true);
        // g.0.set_var(g.1.map(|y| (y-x[0]).powf(2.) / (g.1.nrows() - 1) as f64 ).sum());
        let grad = (-1.)*g.0.grad((&g.1).into(), None);
        println!("grad = {}", grad);
        grad
    };
    let obj = |x : &DVector<f64>, g : &mut (Normal, DMatrix<f64>)| -> f64 {
        println!("param = {}", x);
        // During optimization, we will receive a vector of size one from the optimizer.
        // We must propagate to the actual natural parameter size (3).
        let x = if x.nrows() == 1 {
            DVector::from_element(g.1.nrows(), x[0])
        } else {
            x.clone_owned()
        };
        g.0.set_parameter((&x).into(), true);
        // g.0.set_var(g.1.map(|y| (y-x[0]).powf(2.) / (g.1.nrows() - 1) as f64 ).sum());
        let min = (-1.) * g.0.log_prob((&g.1).into(), None);
        println!("min = {}", min);
        min
    };
    let mut optim = LBFGS::prepare(param, (distr, data))
        .with_gradient(grad)
        .with_function(obj);
    let min = optim.minimize()
        .map_err(|e| format!("Minimization failed: {:?}", e) )?;
    println!("Minimum = {}", min);
    let mut distr = optim.take_data().0;
    distr.set_parameter((&min.value).into(), true);
    Ok(distr)
}

fn mle_compare_univ<D>(distr : D, data : DMatrix<f64>)
where
    D : Distribution + ExponentialFamily<U1> + Clone + Likelihood<U1>
{
    let distr_out = optimize_mle(distr, data.clone())
        .expect("Optimization failed");
    let optim_mle = distr_out.view_parameter(false)[0];
    println!("optim suff stat: {}", optim_mle);
    let stat_mle = D::mle((&data).into()).unwrap().mean()[0];
    println!("suff stat: {}", stat_mle);
    assert!((optim_mle - stat_mle).abs() < 1E-2 );
}

#[test]
fn approx_poiss() {
    let init_lambda = 8.0;
    let poiss = Poisson::new(5, Some(init_lambda));
    let vals = DMatrix::from_column_slice(5, 1, &[1., 2., 3., 1., 2.]);
    mle_compare_univ(poiss, vals);
}

#[test]
fn approx_bern() {
    let init_theta = 0.99;
    let bern = Bernoulli::new(5, Some(init_theta));
    let vals = DMatrix::from_column_slice(5, 1, &[1., 0., 1., 0., 0.]);
    mle_compare_univ(bern, vals);
}

#[test]
fn approx_norm() {
    let init_mu = 0.0;
    let init_sigma = 1.0;

    let vals = DMatrix::from_column_slice(5, 1, &[1.1, 2.2, 3.3, 4.5, 5.5]);
    let mean = vals.sum() / 5.;
    let var = vals.map(|v| (v - mean).powf(2.) ).sum() / 4.;
    let mut norm = Normal::new(5, Some(0.0), Some(var));
    let distr_out = optimize_mle_scaled(norm, vals.clone())
        .expect("Optimization failed");
    let optim_mle = distr_out.view_parameter(false)[0];
    println!("optim suff stat: {}", optim_mle);
    let stat_mle = Normal::mle((&vals).into()).unwrap().mean()[0];
    println!("suff stat: {}", stat_mle);
    assert!((optim_mle - stat_mle).abs() < 1E-2 );
}

fn update_weights<D>(d : &D, w : &mut DMatrix<f64>)
where
    D : Distribution
{
    let var = d.var();
    w.set_diagonal(&var);
}

// Call the iteratively re-weighted least squares algorithm over random y (data).
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
fn irls(mut distr : Bernoulli, y : DMatrix<f64>, x : DMatrix<f64>) -> Result<Bernoulli, String> {
    let mn_init : &MultiNormal = distr.view_factor().unwrap();
    let eta_init = mn_init.view_parameter(true).clone_owned();
    let mut weights = DMatrix::zeros(y.nrows(), y.nrows());
    weights.set_diagonal(&distr.var());

    let mn : &mut MultiNormal = distr.factor_mut().unwrap();
    mn.scale_by(weights.clone() * &x);

    let param = OptimParam::new()
        .init_state(eta_init)
        .preserve(100)
        .max_iter(100);

    // Here, we optimize over scaled_mu (x) but calcualte the gradient wrt the
    // eta vector.
    // Carry (y, X, W).
    let grad = |eta : &DVector<f64>, g : &mut (Bernoulli, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)| -> DVector<f64> {
        let mut w = g.3.clone();
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
        wx.transpose() * (new_lin_eta - old_lin_eta)

        /*// The mean is linearized because mn has the x LinearOp.
        let new_eta = mn.mean().clone_owned();
        // mn.set_parameter((&eta).into(), true);
        let eta_t = DMatrix::from_rows(&[new_eta.transpose()]);
        let grad = (-1.)*mn.grad(eta_t.slice((0, 0), (1, eta_t.ncols())), None);
        // set the new_eta here so the LL is calculated wrt the updated parameter.
        g.0.set_parameter((&new_eta).into(), true);
        println!("grad = {}", grad);
        grad*/
    };
    let obj = |eta : &DVector<f64>, g : &mut (Bernoulli, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)| -> f64 {
        // println!("param = {}", x);
        // During optimization, we will receive a vector of size one from the optimizer.
        // We must propagate to the actual natural parameter size (3).
        // g.0.set_parameter((&eta).into(), true);
        // let mut w = g.2.clone();
        // update_weights(&g.0, &mut w);

        let mn : &MultiNormal = g.0.view_factor().unwrap();
        let y = &g.1;
        let x = &g.2;
        let w = &g.3;
        // let wx = w.clone() * x;
        let wy = (w.clone() * y).transpose();
        // mn.scale_by(wx.clone());
        // (-1.)*mn.log_prob(wy.slice((0, 0), (1, wy.ncols())), None)
        (-1.)*g.0.log_prob(((w.clone() * y)).slice((0, 0), (y.nrows(), 1)), None)

        /*
        mn.set_parameter((&eta).into(), true);
        let new_eta = mn.mean().clone_owned();
        g.0.set_parameter((&new_eta).into(), true);

        // g.0.set_var(g.1.map(|y| (y-x[0]).powf(2.) / (g.1.nrows() - 1) as f64 ).sum());
        let min = (-1.) * g.0.log_prob((&g.1).into(), None);
        println!("min = {}", min);
        min*/
    };
    let mut optim = LBFGS::prepare(param, (distr, y, x, weights))
        .with_gradient(grad)
        .with_function(obj);
    let min = optim.minimize()
        .map_err(|e| format!("Minimization failed: {:?}", e) )?;
    println!("Minimum = {}", min);
    let mut distr = optim.take_data().0;
    distr.set_parameter((&min.value).into(), true);
    Ok(distr)
}

#[test]
fn logistic() {
    let mut bern = Bernoulli::new(100, Some(0.5));
    let norm = Normal::new(100, None, None);
    let mut mn = multinormal::MultiNormal::new_standard(3);
    let x = DMatrix::from_columns(&[
        DVector::from_element(100, 1.),
        norm.sample().column(0).clone_owned(),
        norm.sample().column(0).clone_owned()
    ]);
    let y = bern.sample();
    let bern = bern.condition(mn);
    println!("y = {}", y);
    println!("x = {}", x);
    println!("{:?}", irls(bern, y, x));
}



