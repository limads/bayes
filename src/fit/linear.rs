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
fn test_ols() {

    ols_py = r#"
        import statsmodels.api as sm;
        sm.OLS([1, 1.4, 2.1, 2.4, 3.1], [[1.0, 1.0], [1.0, 1.5], [1.0, 2.0], [1.0, 2.5], [1.0, 3.0]]).fit().summary()
    "#;

    let y : DVector<f64> = DVector::from_vec(vec![1.0, 1.4, 2.1, 2.4, 3.1]);
    let x : DMatrix<f64> = DMatrix::from_vec(5,2,
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0]
    );
    let ols = OLS::estimate_from_data(&y, &x);
    println!("beta = {}", ols.beta);
    println!("err = {}", ols.err.unwrap());
}

/// Carry which range of variables in the data rows are to be considered fixed or random.
pub struct OLSSettings {
    fixed : Range<usize>,
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

impl WLS {

    pub fn estimate_from_cov_diag<S>(
        y : &Matrix<f64, Dynamic, U1, S>,
        sigma_diag : &DVector<f64>,
        x : &DMatrix<f64>
    ) -> Self
    where
        S : Storage<f64, Dynamic, U1>
    {
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

