use nalgebra::*;
use nalgebra::storage::*;
use crate::prob::*;

/// Ordinary least square estimation. This estimator simply solves the linear system X^T X b = X^T y
/// using QR decomposition. It is useful if your have univariate homoscedastic observations conditional
/// on a set of linear predictors, and you don't have prior information to guide inference. If you do
/// (from a previous experiment) the Bayesian regression problem can be solved via the pseudo-data approach:
/// Add the prior mean as a (weighted) pseudo-observation and call the same OLS procedure.
#[derive(Debug)]
pub struct OLS {
    pub beta : DVector<f64>,
    pub err : Option<DVector<f64>>
}

impl OLS {

    /// Builds the bayesian OLS problem, using the distribution prior data. 
    /// The only admissible distribution is the Normal, conditional on a MultiNormal.
    pub fn new<D>(d : D) -> Option<Self> {
        unimplemented!()
    }
    
    /// Carry estimation based on a prediction vector an a cross-products matrix.
    pub fn estimate_from_cp(xy : &DVector<f64>, xx : &DMatrix<f64>) -> Self {
        let xx_qr = xx.clone().qr();
        let beta = xx_qr.solve(&xy).unwrap();
        Self { beta, err : None }
    }

    pub fn estimate(y : &DVector<f64>, x : &DMatrix<f64>) -> Self {
        let xx = x.clone().transpose() * x;
        let xy = x.clone().transpose() * y;
        let mut est = Self::estimate_from_cp(&xy, &xx);
        let err = (x.clone() * &est.beta) - y;
        est.err = Some(err);
        est
    }

}

/// Weighted Least squares algorithm, which estimates
/// the minimum squared error estimate weighting each
/// sample by its corresponding entry in the inverse-diagonal
/// covariance (diagonal precision). This algorithm just the OLS estimator
/// applied to the transformed variables X* = X^T W X and y* = X^T W y, so it is
/// useful if you have heteroscedastic observations conditional on a set of linear
/// predictors, and you don't have informaiton to guide inference.
#[derive(Debug)]
pub struct WLS {

    pub ols: OLS,

    prec_diag : DVector<f64>,

    //err : DVector<f64>
}

impl WLS {

    pub fn estimate_from_cov(
        y : &DVector<f64>,
        cov_diag : &DVector<f64>,
        x : &DMatrix<f64>
    ) -> Self {
        let prec_diag = cov_diag.map(|c| 1. / c );
        Self::estimate_from_prec(&y, &prec_diag, &x)
    }

    pub fn estimate_from_prec(
        y : &DVector<f64>,
        prec_diag : &DVector<f64>,
        x : &DMatrix<f64>
    ) -> Self {
        let prec = DMatrix::<f64>::from_diagonal(&prec_diag);
        let xwx = x.clone().transpose() * &prec * x;
        let xwy = x.clone().transpose() * &prec * y;
        let ols = OLS::estimate_from_cp(&xwy, &xwx);
        Self{ ols, prec_diag : prec_diag.clone() }
    }

}

#[test]
fn test_ols() {
    let y : DVector<f64> = DVector::from_vec(vec![1.0, 1.4, 2.1, 2.4, 3.1]);
    let x : DMatrix<f64> = DMatrix::from_vec(5,2,
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0]
    );
    let ols = OLS::estimate(&y, &x);
    println!("beta = {}", ols.beta);
    println!("err = {}", ols.err.unwrap());
}

/*/// Generalized Least squares algorithm. GLS generalizes WLS for
/// a specified non-diagonal covariance matrix of the **observations** (estimation under any error covariance,
/// such as when observations are temporally or spatially correlated, which leads to an observation block-diagonal
/// matrix). This covariance has an n x n structure and represents the possible
/// correlations of all univariate observations with themselves. The solution is:
/// Multiply X and y by the Cholesky factor of this observation covariance matrix; 
/// This will result in a diagonal observatino matrix, which gets us to a point for which we can solve the 
/// problem via OLS. To make predictions, we pre-multiply any generated quantities by the Cholesky factor again.
#[derive(Debug)]
pub struct GLS {

    pub ols : OLS,

    /// Lower cholesky factor of precision
    chol_prec : DMatrix<f64>

    // cov : DMatrix<f64>,

    // pub x_star : DMatrix<f64>,

    // y_star : DVector<f64>,

    // low_inv : DMatrix<f64>

}

impl GLS {

    fn chol_fact(sigma : DMatrix<f64>) -> Result<DMatrix<f64>, &'static str> {
        let sigma_chol = Cholesky::new(sigma)
            .ok_or("Impossible to calculate cholesky decomposition of covariance matrix")?;
        let low = sigma_chol.l();
        let low_inv = QR::new(low).try_inverse()
            .ok_or("Not possible to invert lower-triangular Cholesky factor of covariance matrix")?;
        Ok(low_inv)
    }

    pub fn estimate_from_prec(
        y : DVector<f64>,
        prec : DMatrix<f64>,
        x : DMatrix<f64>
    ) -> Self {

    }

    pub fn estimate_from_cov(
        y : DVector<f64>,
        cov : DMatrix<f64>,
        x : DMatrix<f64>
    ) -> Self {
        let x = x.unwrap_or(DMatrix::from_element(y.nrows(), 1, 1.));
        let chol_prec = Self::chol_fact(cov.clone()).unwrap();
        let x_star = (chol_prec.clone() * x.clone().transpose()).transpose();
        let y_star = (chol_prec.clone() * y.clone().transpose()).transpose();
        let ols = OLS::estimate(&y_star, &x_star)?;
        Ok(Self { ols, cov, x_star, y_star, low_inv })
    }
}*/

/// The iteratively-reweighted least squares estimator recursively calculates the weighted
/// least squares solution to (y - ybar, X), using var(y bar) as the weights. This estimator
/// generalizes the WLS procedure to non-normal errors, and is widely used for maximum likelihood
/// estimation in logistic and poison regression problems. The resulting distribution represents a lower-bound on the
/// estimator covariance (the Cramer-Rao lower bound), and as such it might underestimate the error
/// of the observations. Importance sampling of the resulting distribution is a relatively cheap fully
/// bayesian follow-up procedure, which informs how severe this underestimation is.
pub struct IRLS {

}

impl IRLS {

    /*pub fn new<D>(distr : impl Distribution) {
        for i in 0..100 {
            let wls = WLS::estimate_from_cov(y, x);
        }
    }*/
    
}
