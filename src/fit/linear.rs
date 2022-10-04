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
#[derive(Debug, Clone)]
pub struct OLS {
    pub beta : DVector<f64>,

    // Inverse matrix of squares and cross-products, (X^T X)^-1. This is the (unnormalized)
    // covariance for OLS; or the CRLB for WLS problems.
    pub sigma_b : DMatrix<f64>,

    pub err : Option<DVector<f64>>,

    pub r_squared : Option<f64>

}

impl OLS {

    /*// R^2 = 1 - (xi-xhat)^2 / (xi-xbar)
    pub fn r_squared(&self, y : &DVector<f64>, x : &DMatrix<f64>) -> Option<f64> {
        let yhat = self.predict(x);
        let ybar = y.mean();
        let rhat = 1. - ()
    }*/

    pub fn predict(&self, x : &DMatrix<f64>) -> DVector<f64> {
        assert!(x.ncols() == self.beta.nrows());
        x.clone() * &self.beta
    }

    /// Carry estimation based on a prediction by solving the linear system of cross-product
    /// matrices (X^T X) and the fixed predictor x observation vector (X^T y) (i.e. normal equations).
    /// This gives the least squares solution b = (X^T X)^{-1} X^T y via QR decomposition. This instantiates
    /// self with only a beta vector, without the error vector.
    pub fn estimate_from_cp(xx : &DMatrix<f64>, xy : &DVector<f64>) -> Option<Self> {
        let xx_qr = xx.clone().qr();
        let beta = xx_qr.solve(&xy)?;
        let sigma_b = xx_qr.try_inverse()?;
        Some(Self { beta, sigma_b, err : None, r_squared : None })
    }

    // estimate from a column of y values and a row-major slice x.chunks(nrows)
    pub fn estimate_from_rows<'a>(
        y : &[f64],
        xs : impl IntoIterator<Item=&'a [f64]> + Clone, 
        add_intercept : bool
    ) -> Option<Self> {
        let ncols = xs.clone().into_iter().next()?.len();
        let n = y.len();
        let dm = if add_intercept {
            let mut dm = DMatrix::zeros(n, ncols + 1);
            dm.column_mut(0).fill(1.0);
            for (i, c) in xs.into_iter().enumerate() {
                dm.row_mut(i).columns_mut(1, ncols).copy_from_slice(c);
            }
            dm
        } else {
            let mut dm = DMatrix::zeros(n, ncols);
            for (i, c) in xs.into_iter().enumerate() {
                dm.row_mut(i).copy_from_slice(c);
            }
            dm
        };
        Self::estimate_from_data(&DVector::from(y.to_vec()), &dm)
    }

    // estimate from a column of y values and a column-major slice x.chunks(ncols)
    pub fn estimate_from_cols<'a>(
        y : &[f64], 
        xs : impl IntoIterator<Item=&'a [f64]> + Clone, 
        add_intercept : bool
    ) -> Option<Self> {
        let ncols = xs.clone().into_iter().count();
        let n = y.len();
        let dm = if add_intercept {
            let mut dm = DMatrix::zeros(n, ncols + 1);
            dm.column_mut(0).fill(1.0);
            for (i, c) in xs.into_iter().enumerate() {
                dm.column_mut(i+1).copy_from_slice(c);
            }
            dm
        } else {
            let mut dm = DMatrix::zeros(n, ncols);
            for (i, c) in xs.into_iter().enumerate() {
                dm.column_mut(i).copy_from_slice(c);
            }
            dm
        };
        Self::estimate_from_data(&DVector::from(y.to_vec()), &dm)
    }
    
    pub fn estimate_from_data(y : &DVector<f64>, x : &DMatrix<f64>) -> Option<Self> {
        let xx = x.clone().transpose() * x;
        let xy = x.clone().transpose() * y;
        let mut ols = Self::estimate_from_cp(&xx, &xy)?;
        let yhat = ols.predict(&x);
        let ybar = y.mean();
        let err = yhat - y;
        let ss_res = err.map(|e| e.powf(2.) ).sum();
        let ss_tot = y.map(|y| (y - ybar).powf(2.) ).sum();
        let r_squared = 1. - (ss_res / (ss_tot+std::f64::EPSILON) );

        // assert!(r_squared >= 0. && r_squared <= 1.);

        ols.err = Some(err);
        ols.r_squared = Some(r_squared);
        Some(ols)
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
    // println!("beta = {}", wls.ols.beta);
    // println!("err = {}", wls.ols.err.unwrap());
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
    let ols = OLS::estimate_from_data(&y, &x).unwrap();
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
        Self { fixed : (1..2) }
    }

    pub fn fixed(mut self, range : Range<usize>) -> Self {
        self.fixed = range;
        self
    }
}

/// Collect first sample of the data iterator to a vector
/// and the remaining samples to a matrix.
fn collect_to_matrix_and_vec(
    sample : impl Iterator<Item=impl Borrow<[f64]>> + Clone
) -> Option<(DVector<f64>, DMatrix<f64>)> {
    let ncol = sample.clone().next()?.borrow().len();

    // Assume at least one dependent and one independent variable.
    if ncol < 2 {
        return None;
    }

    let n = sample.clone().count();
    if n == 0 {
        return None;
    }

    // TODO calc y1..yn if fixed > 1.
    let y = DVector::from_iterator(n, sample.clone().map(|r| r.borrow()[0] ));
    // let x = DMatrix::from_iterator(n, row_len - 1, sample.clone().map(|r| r.borrow()[1..].iter().cloned().collect::<Vec<_>>() ).flatten());
    let mut x = DMatrix::zeros(n, ncol - 1);
    for (ix, row) in sample.enumerate() {
        let fix_vars_row = &row.borrow()[1..];
        assert!(fix_vars_row.len() == ncol - 1);
        x.row_mut(ix).copy_from_slice(&fix_vars_row);
    }
    Some((y, x))
}

impl Estimator for OLS {

    type Settings = OLSSettings;

    type Error = String;

    fn estimate(
        sample : impl Iterator<Item=impl Borrow<[f64]>> + Clone,
        _settings : Self::Settings
    ) -> Result<Self, Self::Error> {
        let (y, x) = collect_to_matrix_and_vec(sample).ok_or(String::from("Invalid data matrix"))?;
        Ok(OLS::estimate_from_data(&y, &x).ok_or(format!("Impossible to solve system"))?)
    }

}

/// Weighted Least squares algorithm, which estimates
/// the minimum squared error estimate weighting each
/// sample by its corresponding entry from a inverse-diagonal
/// covariance (diagonal precision). This algorithm just the OLS estimator
/// applied to the transformed variables X* = W^{1/2} X and y* = W y, which resolves into
/// (X^T W X)^-1 X^T W y, so it is useful if you have heteroscedastic observations
/// conditional on a set of linear predictors, and you don't have informaiton to guide inferencerive
#[derive(Debug, Clone)]
pub struct WLS {

    pub ols: OLS,

    // Precision matrix: Inverse covariance of the individual observations
    sigma_inv : Option<DMatrix<f64>>

}

pub struct WLSSettings {
    fixed : Range<usize>,
    precisions : Option<DVector<f64>>
}

impl WLSSettings {

    pub fn new() -> Self {
        Self { fixed : (1..2), precisions : None }
    }

    pub fn precisions(mut self, precisions : DVector<f64>) -> Self {
        self.precisions = Some(precisions);
        self
    }

    pub fn variances(mut self, mut variances : DVector<f64>) -> Self {
        self.precisions = Some(DVector::from_iterator(variances.len(), variances.iter().map(|v| 1. / *v )));
        self
    }

    pub fn fixed(mut self, range : Range<usize>) -> Self {
        self.fixed = range;
        self
    }
}

impl Estimator for WLS {

    type Settings = WLSSettings;

    type Error = String;

    fn estimate(
        sample : impl Iterator<Item=impl Borrow<[f64]>> + Clone,
        settings : Self::Settings
    ) -> Result<Self, Self::Error> {
        let (y, x) = collect_to_matrix_and_vec(sample).ok_or(String::from("Invalid data matrix"))?;
        let n = y.nrows();
        let prec_diag = settings.precisions.unwrap_or(DVector::repeat(n, 1.));
        Ok(WLS::estimate_from_prec_diag(&y, &prec_diag, &x).ok_or(format!("Impossible to solve system"))?)
    }

}

fn assert_nonzero<'a>(a : impl Iterator<Item=&'a f64>) {
    a.enumerate().for_each(|(ix, it)| assert!(*it != 0.0, "Zero element at position {}", ix) )
}

impl WLS {

    pub fn estimate_from_cov_diag<S>(
        y : &Matrix<f64, Dynamic, U1, S>,
        sigma_diag : &DVector<f64>,
        x : &DMatrix<f64>
    ) -> Option<Self>
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
    ) -> Option<Self>
    where
        S : Storage<f64, Dynamic, U1>
    {
        debug_assert!(is_approx_diagonal(&sigma_inv));
        let xwx = x.clone().transpose() * sigma_inv * x;
        let xwy = x.clone().transpose() * sigma_inv * y;
        let mut ols = OLS::estimate_from_cp(&xwx, &xwy)?;
        ols.err = Some(ols.predict(&x) - y);
        Some(Self{ ols, sigma_inv : Some(sigma_inv.clone_owned()) })
    }

    /// Solves the weighted least squares problem from a vector of diagonal precision values
    /// in a vector.
    pub fn estimate_from_prec_diag<S>(
        y : &Matrix<f64, Dynamic, U1, S>,
        sigma_inv_diag : &DVector<f64>,
        x : &DMatrix<f64>
    ) -> Option<Self>
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
    ) -> bool
    where
        S : Storage<f64, Dynamic, U1>
    {
        let k = self.sigma_inv.as_ref().unwrap().nrows();
        assert!(sigma_inv_diag.nrows() == k);
        for i in 0..k {
            self.sigma_inv.as_mut().unwrap()[(i, i)] = sigma_inv_diag[i];
        }
        if let Some(new) = Self::estimate_from_prec(&y, self.sigma_inv.as_ref().unwrap(), &x) {
            *self = new;
            true
        } else {
            false
        }
    }

    pub fn update_from_cov<S>(
        &mut self,
        y : &Matrix<f64, Dynamic, U1, S>,
        sigma_diag : &DVector<f64>,
        x : &DMatrix<f64>
    ) -> bool where
        S : Storage<f64, Dynamic, U1>
    {
        assert_nonzero(sigma_diag.iter());
        let sigma_inv_diag = sigma_diag.map(|c| 1. / c );
        self.update_from_prec(y, &sigma_inv_diag, x)
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

/*/// Solves the Bayesian least squares problem, when we have a prior (b0, Sb0) for the
/// beta coefficient and the sigma covariance matrix of the regression coefficients.
/// Ridge regression arises as a special case when b_0 = 0 and Sb0 = sigma_b0 I
/// (homoscedastic regression coefficient vector) and Sy = sigma_y (homoscedastic error vector)
/// where lambda = sigma_y / sigma_b0; \lambda \in [0, 1] as (lambda I + X^T X)^-1 X^T y.
/// In either case, this gives a MAP estimate of regression coefficients.
/// (cf. Theodoridis (2020) p. 598).
pub struct BLS {

    b_prior : DVector<f64>,

    sigma_b_prior : DMatrix<f64>,

    // ols.beta will carry the mean of the posterior;
    // ols.sigma_b will carry the covariance of the posterior.
    ols : Option<OLS>,

}

impl BLS {

    pub fn new(
        b_prior : &DVector<f64>,
        sigma_b_prior : &DMatrix<f64>
    ) -> Self {
        Self { b_pior : b_prior.clone(), sigma_b_prior : sigma_b_prior.clone(), ols : None }
    }

    pub fn estimate_from_data(
        &mut self,
        y : &DVector<f64>,
        x : &DMatrix<f64>
    ) -> DVector<f64> {
        self.b_prior +
    }

}*/

/*/// Solves the weighted BLS problem, where we have a prior as in the BLS case,
/// but the observations are not homoscedastic.
pub struct WBLS {

    b_prior : DVector<f64>,

    sigma_b_prior : DMatrix<f64>,

    wls : Option<WLS>

}

impl WBLS {

    pub fn new(
        b_prior : &DVector<f64>,
        sigma_b_prior : &DMatrix<f64>
    ) -> Self {
        Self { b_pior, sigma_b_prior, wls : None }
    }

    /// sigma_inv : obsevation precision (n x n)
    /// b_prior : Coefficient mean prior (p)
    /// sigma_b_prior : Coefficient covariance prior (p x p)
    pub fn estimate_from_data(
        &mut self,
        y : &DVector<f64>,
        sigma_inv : &DMatrix<f64>,
        x : &DMatrix<f64>,
    ) -> Option<Self> {
        // cf. (12.10) of Theodoridis (2020) p. 598.
        let xx_b = sigma_b_prior + x.clone().transpose() * sigma_inv * x;
        let xy_b = sigma_inv * (y.clone() - x.clone() * b_prior);
        let mut ols = OLS::estimate_from_cp(&xx_b, &xy_b)?;
        ols.err = Some(ols.predict(&x) - y);
        Some(Self { b_prior : b_prior.clone(), sigma_b_prior : sigma_b_prior.clone(), ols })
    }

}*/

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

        let wls = WLS::estimate_from_prec(&z, &w, &x).ok_or(format!("Unable to solve system"))?;

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
        let (y, x) = collect_to_matrix_and_vec(sample).ok_or(String::from("Invalid data matrix"))?;
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

/// The Weiner filter is basically a least-squares problem, where the k past
/// samples are taken to be predictors for the next sample. The disadvantage
/// is that we must re-calculate the OLS estimate for each new circular
/// change in the input vector.
pub struct Weiner(OLS);

/// Start with the MSE \hat MSE = arg min_\theta E[(y - \tilde y)^2].
/// Suppose E[\theta] = 0 (prior over \theta is zero) and E[x] = 0
/// (expected value of observation is zero in principle, i.e. x is an error
/// between a noisy and "true" signal). If \theta = A^T X where A is an linear
/// estimator that reduces the MSE, then MSE(A) = E[||\theta - A^T x||_2^2]
/// Leads to the estimate \hat A = \Sigma_{xx}^{-1} \Sigma_{x\theta}
/// and \hat \theta_{LMS} = \Sigma_{x\theta} \Sigma_{xx}^{-1} X (Weiner filter).
/// (derived from Wiener-Hopf Equation).
/// The LMS adaptively updates filter weights (b) by following
/// the steepest-descent direction of the mean squared error
/// between a desired and actual measurement, wrt. the parameter
/// vector. In the long-run, the LMS converges to the Weiner filter.
/// While the algorithm supplies the steepest-descent direction for
/// a new sample, the user must tune the learning rate by informing
/// a constant number in [0.0..1.0].
pub struct LMS {

    // Coefficient vector
    b : DVector<f64>,

    // Actual past output value of dimension 1x1
    y_old : f64,

    // Predictor values of current sample of dimension kx1
    scaled_x_new : DVector<f64>,

    // Learning rate
    rate : f64
}

impl LMS {

    pub fn init(k : usize, rate : f64) -> Self {
        let b = DVector::from_element(k, 1e-6);
        Self { b, y_old : 0.0, scaled_x_new : DVector::zeros(k), rate }
    }

    pub fn step(&mut self, x_new : &[f64], y_new : &[f64]) {
        let x_new = DVectorSlice::from(x_new);
        let err = self.y_old - self.b.dot(&x_new);
        self.scaled_x_new.copy_from(&x_new);
        self.scaled_x_new.scale_mut(err * self.rate);

        // b_new = b_old * rate*err*x_new
        self.b += &self.scaled_x_new;

        self.y_old = self.b.dot(&x_new);

    }

}

pub struct RLS {

    // Coefficient vector
    b : DVector<f64>,

    // P matrix
    p_mat : DMatrix<f64>,

    y_old : f64,

    // Learning rate
    rate : f64,

    z : DVector<f64>,

    scaled_err : DVector<f64>,

    kalman : DVector<f64>
}

impl RLS {

    /// lambda should be a value close to 1.0 (e.g. 0.98, 0.99).
    pub fn init(k : usize, lambda : f64, rate : f64) -> Self {
        let mut p_mat = DMatrix::zeros(k, k);
        p_mat.fill_diagonal(1. / lambda);
        let b = DVector::from_element(k, 1e-6);
        let y_old = 0.0;
        let kalman = DVector::zeros(k);
        let scaled_err = DVector::zeros(k);
        let z = DVector::zeros(k);
        Self { b, p_mat, y_old, rate, scaled_err, kalman, z }
    }

    pub fn step(&mut self, x_new : &[f64], y_new : f64) {
        let x_new = DVectorSlice::from(x_new);

        // Calculate Kalman gain from inverse-covariance and new predictors
        self.z = &self.p_mat * &x_new;
        self.kalman = self.z.clone();
        self.kalman.scale_mut(1. / (self.rate + x_new.dot(&self.z)));

        // Update coefficients
        self.scaled_err = self.kalman.clone();
        let err = self.y_old - self.b.dot(&x_new);
        self.scaled_err.scale_mut(err);
        self.b += &self.scaled_err;

        // Update p matrix
        let mut kal_prod_zt = self.kalman.clone() * self.z.transpose();
        kal_prod_zt.scale_mut(1. / self.rate);
        self.p_mat.scale_mut(1. / self.rate);
        self.p_mat -= &kal_prod_zt;

        self.y_old = self.b.dot(&x_new);
    }
}
