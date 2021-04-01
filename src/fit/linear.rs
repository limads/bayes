use nalgebra::*;
use nalgebra::storage::*;
use crate::prob::*;
use crate::fit::Estimator;
use crate::sample::Sample;
use std::default::Default;

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
    pub err : Option<DVector<f64>>,
    // y : &'a DVector<f64>,
    // x : &'a DVector<f64>,
    // y_pred : Option<DVector<f64>>
}

impl OLS {
    
    /// Carry estimation based on a prediction by solving the linear system of cross-product
    /// matrices (X^T X) and the fixed predictor x observation vector (X^T y). This gives
    /// the least squares solution b = (X^T X)^{-1} X^T y via QR decomposition.
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

    /*pub fn with_residuals(mut self) -> Self {
        let y_pred = self.x.clone() * self.beta;
        self.err = self.y_pred - self.y
    }*/

    pub fn error(&self) -> Option<DVector<f64>> {
        unimplemented!()
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

    // prec_diag : DVector<f64>,

    // Precision matrix: Inverse covariance of the individual observations
    prec : Option<DMatrix<f64>>

    //err : DVector<f64>
}

impl WLS {

    pub fn estimate_from_cov_diag<S>(
        y : &Matrix<f64, Dynamic, U1, S>,
        cov_diag : &DVector<f64>,
        x : &DMatrix<f64>
    ) -> Self
    where
        S : Storage<f64, Dynamic, U1>
    {
        let prec_diag = cov_diag.map(|c| 1. / c );
        Self::estimate_from_prec_diag(&y, &prec_diag, &x)
    }

    /// Solves the weighted least squares problem by informing the
    /// precision (inverse observation variance) matrix (assumed diagonal).
    pub fn estimate_from_prec<S>(
        y : &Matrix<f64, Dynamic, U1, S>,
        prec : &DMatrix<f64>,
        x : &DMatrix<f64>
    ) -> Self
    where
        S : Storage<f64, Dynamic, U1>
    {
        let xwx = x.clone().transpose() * prec * x;
        let xwy = x.clone().transpose() * prec * y;
        let ols = OLS::estimate_from_cp(&xwy, &xwx);
        Self{ ols, prec : Some(prec.clone_owned()) }
    }

    pub fn estimate_from_prec_diag<S>(
        y : &Matrix<f64, Dynamic, U1, S>,
        prec_diag : &DVector<f64>,
        x : &DMatrix<f64>
    ) -> Self
    where
        S : Storage<f64, Dynamic, U1>
    {
        let prec = DMatrix::<f64>::from_diagonal(&prec_diag);
        Self::estimate_from_prec(&y, &prec, &x)
    }

    pub fn update_from_prec<S>(
        &mut self,
        y : &Matrix<f64, Dynamic, U1, S>,
        prec_diag : &DVector<f64>,
        x : &DMatrix<f64>
    ) {
        // self.prec.set_diagonal(prec_diag);
    }

    pub fn update_from_cov<S>(
        &mut self,
        y : &Matrix<f64, Dynamic, U1, S>,
        prec_diag : &DVector<f64>,
        x : &DMatrix<f64>
    ) {

    }

}

/// Represents a linear estimation problem over continuous homoscedastic variables,
/// solved via ordinary least squares. Representing a regression problem with a
/// probabilistic graph means considering a probability->probability relationship
/// in the generative phase; but a probability->constant relationship on the
/// inference stage, if we do not impose priors on the regression coefficients.
pub struct Regression {
    norm : Normal,
    ols : Option<OLS>,
    post : Option<MultiNormal>
}

impl Regression {

    /// Builds a regression problem. For flat-prior (maximum likelihood) estimation,
    /// pass a plain normal. To use prior information, pass a normal conditioned
    /// on a multinormal with the prior regression coefficients.
    pub fn new<D>(norm : Normal) -> Result<Self, String> {
        Ok(Self{ norm, ols : None, post : None })
    }

}

/*impl Estimator<MultiNormal> for Regression {

    fn take_posterior(mut self) -> Option<MultiNormal> {
        self.post.take()
    }

    fn view_posterior<'a>(&'a self) -> Option<&'a MultiNormal> {
        self.post.as_ref()
    }

    /// Runs the inference algorithm for the informed sample matrix,
    /// returning a reference to the modified model (from which
    /// the posterior information of interest can be retrieved).
    fn fit<'a>(&'a mut self, sample : &'a dyn Sample) -> Result<&'a MultiNormal, &'static str> {
        self.norm.observe_sample(sample);
        let (x, y) = retrieve_regression_data(&self.norm).unwrap();
        let ols = OLS::estimate(&y, &x);
        let y_pred = x.clone() * &ols.beta;
        let mse = (y_pred - &y).map(|err| err.powf(2.)).sum() / (x.nrows() - x.ncols()) as f64;
        let cov = mse * (x.clone().transpose() * &x);

        let eta = x * &ols.beta;
        self.norm.set_parameter(eta.column(0).rows(0, eta.nrows()), true);
        let post = MultiNormal::new(ols.beta.len(), ols.beta.clone(), cov).unwrap();
        self.ols = Some(ols);
        self.post = Some(post);
        Ok(self.post.as_ref().unwrap())
    }

}*/

/*impl Predictive for Regression {

    fn predict<'a>(&'a mut self, fixed : Option<&dyn Sample>) -> Option<&'a dyn Sample> {
        if let Some(sample) = fixed {
            self.norm.observe_sample(sample);
        }
        self.norm.predict(None)
    }

    fn view_prediction<'a>(&'a self) -> Option<&'a dyn Sample> {
        self.norm.view_prediction()
    }

}*/

/// Generalized linear model
pub struct GeneralizedRegression<D>
where
    D : Distribution
{
    wls : WLS,
    distr : D
}

pub type PoissonRegression = GeneralizedRegression<Poisson>;

pub type LogisticRegression = GeneralizedRegression<Bernoulli>;

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

pub struct IRLSConfig {
    // Maximum number of iterations
    max_iter : usize,

    // Magnitude of the difference between coefficient vector across two iterations
    tol : f64
}

impl Default for IRLSConfig {
    fn default() -> Self {
        IRLSConfig{ max_iter : 1000, tol : 1E-6 }
    }
}

/*/// The iteratively-reweighted least squares estimator recursively calculates the weighted
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
    
}

impl Estimator<MultiNormal> for IRLS {

    /// Runs the inference algorithm for the informed sample matrix,
    /// returning a reference to the modified model (from which
    /// the posterior information of interest can be retrieved).
    fn fit<'a>(&'a mut self, sample : &'a dyn Sample) -> Result<&'a MultiNormal, &'static str> {
        assert!(self.lik.view_variables().map(|vars| vars.len() ) == Some(1));
        assert!(self.cfg.tol < 1.0);

        self.lik.observe_sample(sample);
        let n_fixed = self.lik.view_fixed().expect("No fixed variables").len();

        let mut eta = DVector::zeros(self.lik.view_variable_values().unwrap().len());
        let mut err = eta.clone();
        let mut n_iter = 0;

        let (x, y) = retrieve_regression_data(self.lik.as_ref()).unwrap();
        let n = y.nrows();

        // Stores coefficients of the current and next iterations
        let mut beta = DVector::from_element(n_fixed, 1.0);
        // let mut beta_next = DVector::from_element(n_fixed, 1.5);

        // Stores the difference between coefficients of the next and current iterations
        // let mut beta_diff = beta.clone();

        let mut w = DMatrix::zeros(n, n);
        let mut near_minimum = false;

        // The Newton-Rhapson equations (expressed as a WLS problem) are:
        // b_{i+1} = b_i + (X^T W X)^-1 X^T (y - \hat y)
        // Which is just an instance of the weighted least squares problem:
        // (X^T W X)^-1 X^T e, for e = (y - \hat y) and W = diag{ 1/ var(y_hat) }
        while !near_minimum && n_iter <= self.cfg.max_iter {
            eta = &x * &beta;
            self.lik.set_parameter(eta.rows(0, eta.nrows()), true);

            let y_pred = self.lik.mean();
            let y_var = self.lik.var();

            // Update weight matrix: w^-1 = (d_eta/d_mu)^2 v_0
            for i in 0..n {
                w[(i,i)] = 1. / y_var[i];
            }

            // Calculate adjusted response variable: z = eta + (y - y_pred) * (d_eta/d_mu)
            // From Bishop (2006) pg. 208: z = Xb - W(y - \hat y)
            err = &y - y_pred;
            let z = eta - w.clone_owned() * err;

            let wls = WLS::estimate_from_prec(&z, &w, &x);

            let beta_diff = wls.ols.beta.clone_owned() - &beta;
            near_minimum = beta_diff.norm() < self.cfg.tol;
            beta.copy_from(&wls.ols.beta);
            n_iter += 1;
        }

        match (n_iter < self.cfg.max_iter, near_minimum) {
            (true, true) => {
                eta = &x * &beta;
                self.lik.set_parameter(eta.rows(0, eta.nrows()), true);
                self.ans = Some(MultiNormal::new(1, beta, x.clone().transpose() * x).unwrap());
                Ok(self.ans.as_ref().unwrap())
            },
            (false, _) => {
                Err("Maximum number of iterations reached")
            },
            (_, false) => {
                Err("Minimum tolerance not achieved")
            }
        }

    }

    /// If fit(.) has been called successfully at least once, returns the current state
    /// of the posterior distribution, whithout changing the algorithm state.
    fn view_posterior<'a>(&'a self) -> Option<&'a MultiNormal> {
        None
    }

    fn take_posterior(self) -> Option<MultiNormal> {
        None
    }

}*/

/* Instead of having fixed fields as a part of a Distribution definition, we can implement
Distribution for D and also implement distribution for Regression<D>, where Regression<D>
fixes the natural parameter at the linear combination:
pub struct Regression<D> {
    lik : D,
    fixed : DMatrix<f64>,
    fixed_names : Vec<String>,
    coefs : Either<MultiNormal, DVector<f64>>,
    solution : LinearSolution
}
Where the lik field does not have any priors.

Then, we define:
impl Conditional<MultiNormal> for Regression<D>
To solve any generalized regression problems (with multinormal prior) or just use Regression<D>
to solve the MLE problem.

view_fixed_names() and view_fixed_values() can be generic methods of this structure instead of
being part of the likelihood implementation. The likelihood implementation can return all names/values.

There is a natural interpretation of any regression problem as first assuming a multivariate normal
joint distribution for (eta, x1..xn), where eta = g(theta), a natural parameter transformation. This
transformation just require that later we interpret the observations y1..yn as heteroscedastic.

A multilevel regression problem (which encopasses regression trees, binary decision trees, etc)
then can be build by implementing impl Conditional<Regression<Normal>> for Regression<D>, where
the fixed values at the inner level are combined with the inner coefficients to produce the
coefficients for the next level (the outer level coefficients are also fixed).

The LinearSolution is an enum:

pub enum LinearSolution {
    Ordinary(OLS),
    Weighted(IRLS)
}

Carrying the algorithm output used to calculate the Posterior implementation for regression.

This solution interprets "regression" as a model formulation; not as algorithm. Of course we
can consider only compositions of distributions as the model formulation, and leave "Regression"
as being only a certain kind of algorithm (OLS/WLS/IRLS).
*/

/* Old Estimate generic structure code.
fn predict_from(
        x : &DMatrix<N>,
        b : &Matrix<N, Dynamic, R, VecStorage<N, Dynamic, R>>
    ) -> Matrix<N, Dynamic, R, VecStorage<N, Dynamic, R>>
        where
            SL : Storage<N, R, U1, RStride = U1, CStride = R>,
            SC : Storage<N, R, R, RStride = U1, CStride = R>,
            D : ExponentialFamily<N, R, SL, SC>
    {
        let eta = (x * b).map(|x| f64::from(x) );
        let yh = match D::TRANSF {
            Transformation::Identity => eta,
            transf => transf.apply::<R>(&eta)
        };
        yh.map(|y| N::from(y))
    }

    /// Prediction with intermediate parameter values (useful during optimization)
    fn predict_unoptimized(
        &self,
        b : &Matrix<N, Dynamic, R, VecStorage<N, Dynamic, R>>
    ) -> Matrix<N, Dynamic, R, VecStorage<N, Dynamic, R>>
        where
            SL : Storage<N, R, U1, RStride = U1, CStride = R>,
            SC : Storage<N, R, R, RStride = U1, CStride = R>,
            D : ExponentialFamily<N, R, SL, SC>
    {
        Self::predict_from(&self.x, b)
    }

    /// Prediction that the user calls, after optimization.
    /// Assumes predictor matrix is augmented with intercept.
    pub fn predict(
        &self,
        x : &DMatrix<N>,
    ) -> Result<Matrix<N, Dynamic, R, VecStorage<N, Dynamic, R>>, &'static str>
        where
            SL : Storage<N, R, U1, RStride = U1, CStride = R>,
            SC : Storage<N, R, R, RStride = U1, CStride = R>,
            D : ExponentialFamily<N, R, SL, SC>
    {

        // Number of covariates per dependent response
        let k = self.x.ncols();

        // number of dependent multivariate responses
        let d = self.y.ncols();

        let uv_mismatch = self.y.ncols() == 1 &&
            self.post.loc().nrows() != x.ncols();
        let mv_mismatch = self.y.ncols() > 1 &&
            self.post.loc().nrows() != x.ncols() * self.y.ncols();
        if uv_mismatch || mv_mismatch {
            return Err("Dimension mismatch when trying to predict value");
        }
        // Parameter vector might be packed into a parameter matrix with coefficients
        // for each dependent variable over columns (if multivariate). This will specialize
        // into DVector<_> for univariate estimates and into DMatrix<_> for multivariate estimates.
        let data = self.post.loc().data;
        let ncols = R::try_to_usize().unwrap_or(d);
        let b = Matrix::<N, Dynamic, R, SliceStorage<N, Dynamic, R, _, _>>::from_slice_generic(
            data.as_slice(),
            Dynamic::new(k),
            R::from_usize(ncols)
        );
        // Possibly apply some covariance scaling here if data is multivariate
        let yh = Self::predict_from(x, &b.into());
        Ok(yh)
    }

    pub fn update_post_cov(&mut self)
        where
            SL : Storage<N, R, U1, RStride = U1, CStride = R>,
            SC : Storage<N, R, R, RStride = U1, CStride = R>,
            D : ExponentialFamily<N, R, SL, SC>
    {
        let n = self.y.nrows();
        let k = self.x.ncols();
        let mut w = DMatrix::<N>::from_element(n, n, N::from(0.0));
        let y_hat = self.predict(&self.x.clone()).unwrap();
        let diag = y_hat.clone().map(|p| N::from(D::TRANSF.var_weight(&f64::from(p))) );
        w.set_diagonal(&diag.column(0));
        let hess = self.x.clone().transpose() * w * self.x.clone();
        let cov =  hess.qr().try_inverse().unwrap(); /*ssq **/
        self.post.update_cov(cov).unwrap();
    }

    /*fn update_err_cov(&mut self) -> Result<(), &'static str> {
        let params = utils::pack_param_matrix(&self.post.loc(), self.y.ncols());
        let eta = match D::TRANSF {
            Transformation::Identity => {
                self.x.clone() * params
            },
            transf => {
                let f_y = self.y.map(|y_el| f64::from(y_el));
                let t_y = transf.apply_inv::<R>(&f_y);
                let y = t_y.map(|t| N::from(t));
                self.x.clone() * params
            }
        };
        let cols : Vec<_> = self.y.column_iter().collect();
        let y_mat = DMatrix::from_columns(&cols[..]);
        let err = y_mat - eta;
        self.lin_err.update_from_sample(err.into())?;
        Ok(())
    }*/

    pub fn error(&self) -> Matrix<N, Dynamic, R, VecStorage<N, Dynamic, R>>
        where
            SL : Storage<N, R, U1, RStride = U1, CStride = R>,
            SC : Storage<N, R, R, RStride = U1, CStride = R>,
            D : ExponentialFamily<N, R, SL, SC>,
             //,
            //DefaultAllocator : Allocator<N, R, Dynamic>
    {
        self.predict(&self.x).unwrap() - &self.y
    }

    /*pub fn error_cov<SL, SC>(&self) -> Matrix<N, Dynamic, R, VecStorage<N, Dynamic, R>>
        where
            SL : Storage<N, R, U1, RStride = U1, CStride = R>,
            SC : Storage<N, R, R, RStride = U1, CStride = R>,
            VecStorage<N, Dynamic, R> : Storage<N, Dynamic, R>,
            D : Distribution<N, R, VecStorage<N, Dynamic, R>, VecStorage<N, R, R>>
    {
        let err = self.error();
        let mut err_t = err.transpose();
        err.transpose_to(&mut err_t);
    }*/

    pub fn sum_squares(&self) -> N
        where
            SL : Storage<N, R, U1, RStride = U1, CStride = R>,
            SC : Storage<N, R, R, RStride = U1, CStride = R>,
            D : ExponentialFamily<N, R, SL, SC>
    {
        let n = self.y.nrows() as f64;
        let p = self.x.ncols() as f64;
        let preds = self.predict(&self.x).unwrap();
        let res = preds.clone() - &self.y;
        let mut w_rsqs = DVector::<N>::from_element(res.nrows(), N::from(0.0));
        for ((rs, pr), wr) in res.iter().zip(preds.iter()).zip(w_rsqs.iter_mut()) {
            let rs_sq = f64::from(*rs).powf(2.0);
            let w = (n - p) * D::TRANSF.var_weight(&f64::from(*pr));
            *wr = N::from(rs_sq / w )
        }
        w_rsqs.sum()
    }

    /// -2 log(ll_sat / ll_red) can be shown to be asymptotically X^2
    /// and used in decision analysis.
    /// Move this into decision module later.
    pub fn deviance(&self) -> N
        where
            SL : Storage<N, R, U1, RStride = U1, CStride = R>,
            SC : Storage<N, R, R, RStride = U1, CStride = R>,
            D : ExponentialFamily<N, R, SL, SC>
    {
        let yh = self.predict(&self.x).unwrap();
        let ll_sat = D::log_prob_unscaled(&self.y, &self.y);
        let ll_red = D::log_prob_unscaled(&yh, &self.y);
        N::from(-2.) * (ll_sat.sum() - ll_red.sum())
    }

    /*pub fn aic(&self, other : Self) -> N {
        self.likelihood.log_prob(self.params().loc()) -
            other.likelihood.log_prob(other.params().loc())
    }*/
*/

/*
GSL WLS

pub fn gls_wls(
    y : DVector<f64>,
    x : DMatrix<f64>,
    w : DVector<f64>
) -> Option<WLSResult> {
    let n = y.nrows();
    let p = x.ncols();
    unsafe{
        let ws = gsl_multifit_linear_alloc(n, p);
        let y_gsl : gsl_vector = y.into();
        let x_gsl : gsl_matrix = x.into();
        let w_gsl : gsl_vector = w.into();
        let mut b_gsl : gsl_vector =
            DVector::<f64>::from_element(p, 0.0).into();
        let mut cov_gsl : gsl_matrix =
            DMatrix::<f64>::from_element(p, p, 0.0).into();
        let mut chi_sq : f64 = 0.0;
        let fit_err = gsl_multifit_wlinear(
            &x_gsl as *const _,
            &w_gsl as *const _,
            &y_gsl as *const _,
            &mut b_gsl as *mut _,
            &mut cov_gsl as *mut _,
            &mut chi_sq as *mut _,
            ws
        );
        match fit_err {
            0 => {
                let mut resid_gsl : gsl_vector = DVector::<f64>::from_element(n, 0.0).into();
                let resid_err = gsl_multifit_linear_residuals(
                    &x_gsl as *const _,
                    &y_gsl as *const _,
                    &b_gsl as *const _,
                    &mut resid_gsl as *mut _
                );
                if resid_err != 0 {
                    println!("GSL Error at recovering residuals: {}", resid_err);
                    return None;
                }
                let beta : DVector<f64> = b_gsl.into();
                let cov : DMatrix<f64> = cov_gsl.into();
                let resid : DVector<f64> = resid_gsl.into();
                Some( WLSResult{beta, cov, resid} )
            },
            _ => {
                println!("GSL Error at fitting WLS model: {}", fit_err);
                None
            }
        }
    }
}

Argmin minimization
impl ArgminOp for LogitEstimate {

    type Param = DVector<f64>;

    type Output = f64;

    type Hessian = DMatrix<f64>;

    type Jacobian = DMatrix<f64>;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let n = self.sample.clone().nrows();
        let x = self.preds.clone().unwrap();
        let y = self.sample.clone();
        let y_compl = y.map(|p| 1.0 - p);
        let y_hat = (x * p.clone()).map(|l| (Binomial::<f64>::LINK.unwrap())(&l) );
        let y_hat_log = y_hat.clone().map(|p| p.ln() );
        let y_hat_compl_log = y_hat.map(|p| (1.0 - p).ln() );

        // Log-likelihood scalar, as a function of the parameters.
        let nll = -1. * (y.component_mul(&y_hat_log) +
            y_compl.component_mul(&y_hat_log)).sum();
        Ok(nll)
    }

    /// Compute the gradient at parameter `p`.
    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
        let n = self.sample.nrows();
        //let k = self.preds.unwrap().ncols();
        let x = self.preds.clone().unwrap();
        let y = self.sample.clone();
        let mut y_hat = (x.clone() * p.clone()).map(|l| (Binomial::<f64>::LINK.unwrap())(&l));

        // The jacobian for logistic regression is simply
        // the score vector (one diff for each param).
        let score = x.transpose() * (y - y_hat.clone());

        // let iter_weights = y_hat.map(|p| 1. / (p * (1.0 - p)) );
        // println!("{}", iter_weights);

        // But GSL requires one update row for each sample. We just copy the
        // score to all independent (row) samples.
        /*let mut jac = DMatrix::<f64>::from_element(n, p, 0.0);
        for (i, mut r) in jac.row_iter_mut().enumerate() {
            let row = iter_weights[i] * score.transpose().clone();
            r.copy_from(&row);
        }
        jac*/
        Ok(score)
    }

    /// Compute the Hessian at parameter `p`.
    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        let k = self.preds.clone().unwrap().ncols();
        let y_hat = (self.preds.clone().unwrap().clone() * p)
            .map(|l| (Binomial::<f64>::LINK.unwrap())(&l) );
        let weights_diag = y_hat.map(|p| (p * (1.0 - p)) );
        let mut weights = DMatrix::<f64>::from_element(k, k, 0.0);
        weights.set_diagonal(&weights_diag);
        let x = self.preds.clone().unwrap();
        let hess = -1. * x.clone().transpose() * weights * x;
        Ok(hess)
    }
}*/

