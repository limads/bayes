use nalgebra::*;
use nalgebra::storage::*;
use crate::fit::Estimator;
use std::default::Default;
use std::borrow::Borrow;
use std::ops::Range;

/// Ordinary least square estimation. This estimator simply solves the linear system X^T X b = X^T y
/// using QR decomposition. It is useful if your have univariate homoscedastic observations conditional
/// on a set of fixed linear predictors. If you don't have prior information on the regression coefficients,
/// initialize this estimator with a plain Normal variable with the fixed and random names bound to it.
/// If you do (for example, you want to use data form a previous study with the same design),
/// you can condition this normal on a multivariate normal with the prior values. In this case, the
/// posterior will be found via the pseudo-data approach:
/// Add the prior mean as a (n-weighted) pseudo-observation and call the same OLS procedure.
#[derive(Debug)]
pub struct OLS {
    pub beta : DVector<f64>,

    // Inverse matrix of squares and cross-products, (X^T X)^-1. This is the (unnormalized)
    // covariance for OLS; or the CRLB for WLS problems.
    pub sigma_b : DMatrix<f64>,

    pub err : Option<DVector<f64>>
}

impl OLS {

    pub fn predict(&self, x : &DMatrix<f64>) -> DVector<f64> {
        assert!(x.ncols() == self.beta.nrows());
        x.clone() * &self.beta
    }

    /// Carry estimation based on a prediction by solving the linear system of cross-product
    /// matrices (X^T X) and the fixed predictor x observation vector (X^T y). This gives
    /// the least squares solution b = (X^T X)^{-1} X^T y via QR decomposition. This instantiates
    /// self with only a beta vector, without the error vector.
    pub fn estimate_from_cp(xx : &DMatrix<f64>, xy : &DVector<f64>) -> Self {
        let xx_qr = xx.clone().qr();
        let beta = xx_qr.solve(&xy).unwrap();
        let sigma_b = xx_qr.try_inverse().unwrap();
        Self { beta, sigma_b, err : None }
    }

    pub fn estimate_from_data(y : &DVector<f64>, x : &DMatrix<f64>) -> Self {
        let xx = x.clone().transpose() * x;
        let xy = x.clone().transpose() * y;
        let mut ols = Self::estimate_from_cp(&xx, &xy);
        ols.err = Some(ols.predict(&x) - y);
        ols
    }

}

#[test]
fn test_wls() {
    let wls_py = r#"
        import statsmodels.api as sm;
        sm.WLS(
            [1, 1.4, 2.1, 2.4, 3.1],
            [[1.0, 1.0], [1.0, 1.5], [1.0, 2.0], [1.0, 2.5], [1.0, 3.0]],
            [1./0.1, 1./0.1, 1./0.2, 1./0.3, 1./0.4]
        ).fit().summary()
    "#;

    let y : DVector<f64> = DVector::from_vec(vec![1.0, 1.4, 2.1, 2.4, 3.1]);
    let x : DMatrix<f64> = DMatrix::from_vec(5,2,
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0]
    );
    let cov_diag = DVector::from_vec(vec![0.1, 0.1, 0.2, 0.3, 0.4]);
    let wls = WLS::estimate_from_cov_diag(&y, &cov_diag, &x);
    println!("beta = {}", wls.ols.beta);
    println!("err = {}", wls.ols.err.unwrap());
}

#[test]
fn test_ols() {

    let ols_py = r#"
        import statsmodels.api as sm;
        sm.OLS(
            [1, 1.4, 2.1, 2.4, 3.1],
            [[1.0, 1.0], [1.0, 1.5], [1.0, 2.0], [1.0, 2.5], [1.0, 3.0]]
        ).fit().summary()
    "#;

    let y : DVector<f64> = DVector::from_vec(vec![1.0, 1.4, 2.1, 2.4, 3.1]);
    let x : DMatrix<f64> = DMatrix::from_vec(5,2,
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0]
    );
    let ols = OLS::estimate_from_data(&y, &x);
    println!("beta = {}", ols.beta);
    println!("err = {}", ols.err.unwrap());
}

/// Carry which range of variables in the data rows are to be considered fixed.
/// Variables outside this range are assumed to be random, conditional on the
/// fixed variables. The output will be an array of vectors, each vector corresponding
/// to the kth random variable.
pub struct OLSSettings {
    fixed : Range<usize>,
}

impl OLSSettings {

    pub fn new() -> Self {
        Self { fixed : (1..1) }
    }

    pub fn fixed(mut self, range : Range<usize>) -> Self {
        self.fixed = range;
        self
    }
}

/// Collect first sample of the data iterator to a vector
/// and the remaining samples to a matrix.
fn collect_to_matrix_and_vec(
    sample : impl Iterator<Item=impl Borrow<[f64]>> +
    Clone
) -> (DVector<f64>, DMatrix<f64>) {
    let row_len = sample.clone().next().unwrap().borrow().len();
    let n = sample.clone().count();
    let y = DVector::from_iterator(n, sample.clone().map(|r| r.borrow()[0] ));
    let x = DMatrix::from_iterator(row_len - 1, n, sample.clone().map(|r| r.borrow().iter().cloned().collect::<Vec<_>>() ).flatten());
    (y, x)
}

impl Estimator for OLS {

    type Settings = OLSSettings;

    type Error = ();

    fn estimate(
        sample : impl Iterator<Item=impl Borrow<[f64]>> + Clone,
        _settings : Self::Settings
    ) -> Result<Self, Self::Error> {
        let (y, x) = collect_to_matrix_and_vec(sample);
        Ok(OLS::estimate_from_data(&y, &x))
    }

}

/// Weighted Least squares algorithm, which estimates
/// the minimum squared error estimate weighting each
/// sample by its corresponding entry from a inverse-diagonal
/// covariance (diagonal precision). This algorithm just the OLS estimator
/// applied to the transformed variables X* = W^{1/2} X and y* = W y, which resolves into
/// (X^T W X)^-1 X^T W y, so it is useful if you have heteroscedastic observations
/// conditional on a set of linear predictors, and you don't have informaiton to guide inferencerive
#[derive(Debug)]
pub struct WLS {

    pub ols: OLS,

    // Precision matrix: Inverse covariance of the individual observations
    sigma_inv : Option<DMatrix<f64>>

}

fn assert_nonzero<'a>(a : impl Iterator<Item=&'a f64>) {
    a.enumerate().for_each(|(ix, it)| assert!(*it != 0.0, "Zero element at position {}", ix) )
}

impl WLS {

    pub fn estimate_from_cov_diag<S>(
        y : &Matrix<f64, Dynamic, U1, S>,
        sigma_diag : &DVector<f64>,
        x : &DMatrix<f64>
    ) -> Self
    where
        S : Storage<f64, Dynamic, U1>
    {
        assert_nonzero(sigma_diag.iter());
        let sigma_inv_diag = sigma_diag.map(|c| 1. / c );
        Self::estimate_from_prec_diag(&y, &sigma_inv_diag, &x)
    }

    /// Solves the weighted least squares problem by informing the
    /// precision (inverse observation variance) matrix (assumed diagonal).
    pub fn estimate_from_prec<S>(
        y : &Matrix<f64, Dynamic, U1, S>,
        sigma_inv : &DMatrix<f64>,
        x : &DMatrix<f64>
    ) -> Self
    where
        S : Storage<f64, Dynamic, U1>
    {
        debug_assert!(is_approx_diagonal(&sigma_inv));
        let xwx = x.clone().transpose() * sigma_inv * x;
        let xwy = x.clone().transpose() * sigma_inv * y;
        let mut ols = OLS::estimate_from_cp(&xwx, &xwy);
        ols.err = Some(ols.predict(&x) - y);
        Self{ ols, sigma_inv : Some(sigma_inv.clone_owned()) }
    }

    /// Solves the weighted least squares problem from a vector of diagonal precision values
    /// in a vector.
    pub fn estimate_from_prec_diag<S>(
        y : &Matrix<f64, Dynamic, U1, S>,
        sigma_inv_diag : &DVector<f64>,
        x : &DMatrix<f64>
    ) -> Self
    where
        S : Storage<f64, Dynamic, U1>
    {
        let sigma_inv = DMatrix::<f64>::from_diagonal(&sigma_inv_diag);
        Self::estimate_from_prec(&y, &sigma_inv, &x)
    }

    pub fn update_from_prec<S>(
        &mut self,
        y : &Matrix<f64, Dynamic, U1, S>,
        sigma_inv_diag : &DVector<f64>,
        x : &DMatrix<f64>
    )
    where
        S : Storage<f64, Dynamic, U1>
    {
        let k = self.sigma_inv.as_ref().unwrap().nrows();
        assert!(sigma_inv_diag.nrows() == k);
        for i in 0..k {
            self.sigma_inv.as_mut().unwrap()[(i, i)] = sigma_inv_diag[i];
        }
        *self = Self::estimate_from_prec(&y, self.sigma_inv.as_ref().unwrap(), &x);
    }

    pub fn update_from_cov<S>(
        &mut self,
        y : &Matrix<f64, Dynamic, U1, S>,
        sigma_diag : &DVector<f64>,
        x : &DMatrix<f64>
    ) where
        S : Storage<f64, Dynamic, U1>
    {
        assert_nonzero(sigma_diag.iter());
        let sigma_inv_diag = sigma_diag.map(|c| 1. / c );
        self.update_from_prec(y, &sigma_inv_diag, x);
    }

}

/// Checks that the matrix does not differ sigificantly from a zero (diagonal) matrix.
fn is_approx_diagonal(m : &DMatrix<f64>) -> bool {
    if m.is_square() {
        let n = m.nrows();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    if m[(i, j)].abs() > f64::EPSILON {
                        return false;
                    }
                }
            }
        }
        true
    } else {
        false
    }
}

/// Solves the Bayesian least squares problem, when we have a prior (b0, Sb0) for the
/// beta coefficient and the sigma covariance matrix of the regression coefficients.
/// Ridge regression arises as a special case when b_0 = 0 and Sb0 = sigma_b0 I
/// (homoscedastic regression coefficient vector) and Sy = sigma_y (homoscedastic error vector)
/// where lambda = sigma_y / sigma_b0; \lambda \in [0, 1] as (lambda I + X^T X)^-1 X^T y.
/// In either case, this gives a MAP estimate of regression coefficients.
pub struct BLS {
    b_prior : DVector<f64>,
    sigma_b_prior : DMatrix<f64>,

    // ols.beta will carry the mean of the posterior;
    // ols.sigma_b will carry the covariance of the posterior.
    ols : OLS,

}

impl BLS {

    /// sigma_inv : obsevation precision (n x n)
    /// b_prior : Coefficient mean prior (p)
    /// sigma_b_prior : Coefficient covariance prior (p x p)
    pub fn estimate_from_data(
        y : &DVector<f64>,
        sigma_inv : &DMatrix<f64>,
        x : &DMatrix<f64>,
        b_prior : &DVector<f64>,
        sigma_b_prior : &DMatrix<f64>
    ) -> Self {
        let xy_b = sigma_inv * (y.clone() - x.clone() * b_prior);
        let xx_b = sigma_b_prior + x.clone().transpose() * sigma_inv * x;
        let mut ols = OLS::estimate_from_cp(&xx_b, &xy_b);
        ols.err = Some(ols.predict(&x) - y);
        Self { b_prior : b_prior.clone(), sigma_b_prior : sigma_b_prior.clone(), ols }
    }

}

/// Solves the iteratively-reweighted least squares, using
/// the variance function (a closure that takes the predicted value
/// and returns the variance for this value) using the current estimate
/// as the variance for the weight matrix.
fn var_func_irls(
    y : DVector<f64>,
    x : DMatrix<f64>,
    link : impl Fn(&f64)->f64, // pass Bernoulli::link here
    var : impl Fn(&f64)->f64,  // Pass variance for given mean, p(1-p^2/n) for Bernoulli/Binomial
    tol : f64,
    max_iter : usize
) -> Result<DVector<f64>, String> {
    let (n, p) = (x.nrows(), x.ncols());
    let mut eta = DVector::zeros(n);
    let mut err = eta.clone();
    let mut n_iter = 0;

    // Stores coefficients of the current and next iterations
    let mut beta = DVector::from_element(p, 1.0);

    // Stores the difference between coefficients of the next and current iterations
    // let mut beta_diff = beta.clone();

    let mut w = DMatrix::zeros(n, n);
    let mut near_minimum = false;

    // The Newton-Rhapson equations (expressed as a WLS problem) are:
    // b_{i+1} = b_i + (X^T W X)^-1 X^T (y - \hat y)
    // Which is just an instance of the weighted least squares problem:
    // (X^T W X)^-1 X^T e, for e = (y - \hat y) and W = diag{ 1/ var(y_hat) }
    while !near_minimum && n_iter <= max_iter {
        eta = &x * &beta;

        let y_pred = eta.map(|e| link(&e) );

        // TODO verify if/why variance of observations equal abslute error at y data scale (see irls impl)
        let y_var = y_pred.map(|y| var(&y) );

        // Update weight matrix: w^-1 = (d_eta/d_mu)^2 v_0
        for i in 0..n {
            w[(i,i)] = 1. / y_var[i];
        }

        // Calculate adjusted response variable: z = eta + (y - y_pred) * (d_eta/d_mu)
        // From Bishop (2006) pg. 208: z = Xb - W(y - \hat y)
        err = &y - y_pred;

        // From Bolstad p.183 / McCullagh & Nelder (1983)
        let z = eta + w.clone_owned() * err;

        let wls = WLS::estimate_from_prec(&z, &w, &x);

        let beta_diff = wls.ols.beta.clone_owned() - &beta;
        near_minimum = beta_diff.norm() < tol;
        beta.copy_from(&wls.ols.beta);
        n_iter += 1;
    }

    match (n_iter < max_iter, near_minimum) {
        (true, true) => {
            Ok(beta)
        },
        (false, _) => {
            Err(format!("Maximum number of iterations reached"))
        },
        (_, false) => {
            Err(format!("Minimum tolerance not achieved"))
        }
    }
}

fn update_weights(w : &mut DMatrix<f64>, err : &DVector<f64>) {
    let n = w.nrows();
    for i in 0..n {
        w[(i, i)] = 1. / err[i].max(1E-12);
    }
}

/// Solves the iteratively-reweighted least squares, using
/// the absolute error between predicted values and current estimates
/// as the variance for the weight matrix.
fn abs_err_irls(
    y : DVector<f64>,
    x : DMatrix<f64>,
    link : impl Fn(&f64)->f64, // pass Bernoulli::link here
    tol : f64,
    max_iter : usize
) -> Result<DVector<f64>, String> {
    let (n, p) = (x.nrows(), x.ncols());
    assert!(x.nrows() == y.nrows());
    let mut w = DMatrix::zeros(n, n);
    let mut coefs = DVector::from_element(p, 1.0);
    let mut diff_coefs = DVector::from_iterator(p, (0..p).map(|_| f64::INFINITY ));
    let mut n_iter = 0;

    while (diff_coefs.norm() > tol) && (n_iter <= max_iter) {
        let eta = x.clone_owned() * &coefs;
        let y_pred = eta.map(|e| link(&e) );

        // TODO examine why 1/err can be used to approximate y_pred.variance().
        // For beronulli, variance weights should be 1/phat(1-phat), not 1/(phat - p) as we are doing here.
        let err = (y.clone() - y_pred).abs();
        update_weights(&mut w, &err);

        // Calculate (X^T W X)^{-1}. Note: This step equals
        // solving WLS::estimate_from_cov_diag(x, err, y)
        let squared_prod = (x.clone().transpose() * w.clone() * &x);
        let qr_s = QR::new(squared_prod);
        if let Some(inv_squared_prod) = qr_s.try_inverse() {

            // Calculate (X^T W y)
            let cross_prod = x.clone().transpose() * &w * &y;
            let new_coefs = inv_squared_prod * cross_prod;
            diff_coefs = new_coefs.clone() - &coefs;
            coefs = new_coefs;
            n_iter += 1;

        } else {
            return Err(String::from("Unable to invert square-product matrix"));
        }
    }

    if n_iter <= max_iter {
        println!("IRLS completed (done in {} iterations)", n_iter);
        Ok(coefs)
    } else {
        Err(String::from("Algorithm did not converge"))
    }
}

#[test]
fn test_irls() {
    let ols_py = r#"
        import statsmodels.api as sm;
        sm.GLM(
            [1, 0, 1, 0, 1],
            [[1.0, 1.0], [1.0, 1.5], [1.0, 2.0], [1.0, 2.5], [1.0, 3.0]],
            sm.families.Binomial()
        ).fit().summary()

        # OR

        sm.Logit(
            [1, 0, 1, 0, 1],
            [[1.0, 1.0], [1.0, 1.5], [1.0, 2.0], [1.0, 2.5], [1.0, 3.0]],
        ).fit().summary()

        # OR
        data <- as.data.frame(list(y = c(1, 0, 1, 0, 1), x1 = c(1.0, 1.5, 2.0, 2.5, 3.0)))
        summary(glm("y~1+x1", family=binomial(link="logit"), data))
    "#;
    let y : DVector<f64> = DVector::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0]);
    let x : DMatrix<f64> = DMatrix::from_vec(5,2,
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0]
    );

    // println!("{:?}", abs_err_irls(y, x, crate::calc::Variate::sigmoid, 0.000000000001, 10000) );
    // println!("{:?}", var_func_irls(y, x, crate::calc::Variate::sigmoid, |p : &f64| p*(1. - *p), 0.0001, 10000) );

    // Variance function retrieved from Table1 at https://www.statsmodels.org/stable/glm.html.
    // p - p^2/n = p(1 - p/n) where n is the domain of the corresponding Binomial. For n=1 this does
    // not converge; for n>=2 this converge to the right value.
    println!("{:?}", var_func_irls(y, x, crate::calc::Variate::sigmoid, |p : &f64| *p - p.powf(2.) / 5., 0.0001, 10000) );
}

/// The iteratively-reweighted least squares estimator recursively calculates the weighted
/// least squares solution to (y - ybar, X), using var(y bar) as the weights. This estimator
/// generalizes the WLS procedure to non-normal errors, and is widely used for maximum likelihood
/// estimation in logistic and poison regression problems. The resulting distribution represents a lower-bound on the
/// estimator covariance (the Cramer-Rao lower bound), and as such it might underestimate the error
/// of the observations. Importance sampling of the resulting distribution is a relatively cheap fully
/// bayesian follow-up procedure, which informs how severe this underestimation is.
/// If the informed likelihood has no MultiNormal conditional expectation coefficient prior,
/// the prior is assumed to be uniform, and the posterior covariance will represent the Cramer-Rao Lower Bound for the
/// estimates.
pub struct IRLS {
    settings : IRLSSettings,
    beta : DVector<f64>
}

#[derive(Debug, Clone, Copy)]
pub enum Family {
    Binomial,
    Poison,
    Normal
}

pub struct IRLSSettings {
    fixed : Range<usize>,
    family : Family,
    tol : f64,
    max_iter : usize
}

impl Estimator for IRLS {

    type Settings = IRLSSettings;

    type Error = String;

    fn estimate(
        sample : impl Iterator<Item=impl Borrow<[f64]>> + Clone,
        settings : Self::Settings
    ) -> Result<Self, Self::Error> {
        let (y, x) = collect_to_matrix_and_vec(sample);
        let n = y.nrows();

        let (link, var) : (Box<dyn Fn(&f64)->f64>, Box<dyn Fn(&f64)->f64>) = match settings.family {
            Family::Binomial => (Box::new(crate::calc::Variate::sigmoid), Box::new(move |p : &f64| *p - p.powf(2.) / n as f64)),
            Family::Poison => (Box::new(|v : &f64| v.ln() ), Box::new(crate::calc::Variate::identity)),
            _ => unimplemented!()
        };

        var_func_irls(
            y,
            x,
            link,
            var,
            settings.tol,
            settings.max_iter,
        ).map(|beta| Self { beta, settings })
    }

}

/*pub struct IRLS {
    lik : Box<dyn Likelihood>,

    // Difference between two iterations of the coefficient vector magnitude,
    // used as a criterion to stop the optimization.
    cfg : IRLSConfig,

    ans : Option<MultiNormal>
}

impl IRLS {

    pub fn new<L>(lik : L, config : Option<IRLSConfig>) -> Result<Self, ()>
    where
        L : Likelihood + 'static
    {
        Ok(Self{ lik : Box::new(lik), cfg : config.unwrap_or(Default::default()), ans : None })
    }

    /*// Finds the maximum likelihood estimator.
    fn mle(y : &dyn Sample, x : &dyn Sample) -> Self {
        match
    }*/

    /*// Runs IRLS with the informed prior for the regression coefficients
    fn map<D>(d : D) -> Self {

    }
    */

    /*pub fn new<D>(distr : impl Distribution) {
        for i in 0..100 {
            let wls = WLS::estimate_from_cov(y, x);
        }
    }*/

}*/

