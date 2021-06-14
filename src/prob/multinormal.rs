use nalgebra::*;
use super::*;
use serde::{Serialize, Deserialize};
use std::f64::consts::PI;
use std::fmt::{self, Display};
use std::default::Default;
use std::ops::{SubAssign, MulAssign};
use std::convert::{TryFrom, TryInto};
use crate::fit::markov::Trajectory;
use nalgebra::linalg;
use serde_json::{self, Value, map::Map};
use anyhow;
use crate::model::Model;
use crate::fit;
use crate::fit::Estimator;
use std::iter::FromIterator;
use arraymap::*;

/*#[derive(Debug, Clone, Serialize, Deserialize)]
enum CovFunction {
    None,
    Log,
    Logit
}*/

/// A variable scale means a scale factor that varies from one sample to another
/// (as we have in a multivariate regression setting), represented in a tall data matrix.
/// A constant scale means a scale factor that is the same across all random variable realizations.
enum Scale {
    Variable(DMatrix<f64>),
    Constant(DMatrix<f64>)
}

/// Represents a lazily-evaluated constant linear operator applied to 
/// all realizations of a MultiNormal.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinearOp {

    pub scale : DMatrix<f64>,

    pub scale_t : DMatrix<f64>,

    pub shift : DVector<f64>,

    // pub scale_t : DMatrix<f64>,

    dst : Box<MultiNormal>
}

/*impl Default for LinearOp {

    fn default() -> Self {
        let scale = DMatrix::from_element(1,1,1.);
        let scale_t = scale.clone();
        let shift = DVector::from_element(1, 0.);
        let lin_mu = shift.clone();
        let lin_scaled_mu = shift.clone();
        let lin_sigma_inv = scale.clone();
        //let transf_sigma_inv = None;
        //let cov_func = CovFunction::None;
        Self {
            scale,
            scale_t,
            shift,
            lin_mu,
            lin_sigma_inv,
            lin_scaled_mu,
            lin_log_part : DVector::zeros(1)
            //transf_sigma_inv,
            //cov_func
        }
    }

}*/

impl LinearOp {

    pub fn new(
        src : &MultiNormal,
        shift : Option<DVector<f64>>,
        scale : Option<DMatrix<f64>>
    ) -> Self {
        let n = shift.clone().map(|s| s.nrows()).unwrap_or(scale.clone().unwrap().nrows());
        let create_ident = || {
            let mut m = DMatrix::zeros(n, n);
            m.fill_with_identity();
            m
        };
        let scale = scale.unwrap_or_else(create_ident);
        let scale_t = scale.transpose();
        let shift = shift.unwrap_or_else(|| DVector::from_element(n, 0.));
        let mut m = MultiNormal::new(src.n, shift.clone(), create_ident()).unwrap();
        let mut op = Self{ scale, scale_t, shift, dst : Box::new(m) };
        op.update_from(&src);
        op
    }

    pub fn update_from(&mut self, src : &MultiNormal) {
        let new_mu = self.shift.clone() + &self.scale * src.mean();
        let src_cov = MultiNormal::invert_scale(&src.sigma_inv);
        println!("src sigma = {}", src_cov);
        println!("scale = {}", self.scale);
        println!("scale_t = {}", self.scale_t);
        let new_sigma = self.scale.clone() * src_cov * &self.scale_t;
        self.dst.set_parameter(new_mu.rows(0, new_mu.nrows()), false);
        println!("dst sigma = {}", new_sigma);
        if let Some(cov) = QR::new(new_sigma.clone()).try_inverse() {
            self.dst.set_cov(new_sigma);
        } else {
            panic!("Scaling did not produce valid covariance");
        }
    }

    pub fn update_scale(&mut self, src : &MultiNormal, scale : DMatrix<f64>) {
        // Assert that there cannot be colinearities among scale columns, or repeated rows,
        // or else X S X^T won't be invertible and PD if S is identity.
        assert!(self.scale.nrows() == scale.nrows());
        assert!(self.scale.ncols() == scale.ncols());
        self.scale = scale.clone();
        self.scale_t = scale.transpose();
        let cov = MultiNormal::invert_scale(&src.sigma_inv);
        let new_sigma = self.scale.clone() * cov * &self.scale_t;

        // For linear modelling, the X matrix do not need to yield a valid
        // inversion here.
        if let Some(cov) = QR::new(new_sigma.clone()).try_inverse() {
            self.dst.set_cov(new_sigma);
        } else {
            panic!("Scaling did not produce valid covariance");
        }
    }

    pub fn update_shift(&mut self, src : &MultiNormal, shift : DVector<f64>) {
        assert!(self.shift.nrows() == shift.nrows());
        self.shift = shift;
        let new_mu = self.shift.clone() + self.scale.clone() * src.mean();
        self.dst.set_parameter(new_mu.rows(0, new_mu.nrows()), false);
    }

}

/*
Perhaps use as reference the book "Computational Bayesian Statistics"
https://web.ma.utexas.edu/users/pmueller/compbayes/
*/

/// Multivariate normal parametrized by μ (px1) and Σ (pxp). While the public API always
/// work by receiving covariance matrices, this structure holds both the covariance and
/// the precision (inverse covariance) matrices internally. Re-setting the covariance
/// is potentially a costly operation, since a QR inversion is performed.
///
/// Any graph of joint continuous distributions linked by a correlation coefficient
/// resolves to a single MultiNormal distribution. The graph might have independent child
/// nodes, which are represented by a top-level JSON array. Each element, then, link
/// to an JSON array of parent nodes, whose elements are also distribution implementors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiNormal {

    n : usize,
    
    mu : DVector<f64>,

    /// (sigma_inv * mu)^T
    scaled_mu : DVector<f64>,

    sigma : DMatrix<f64>,

    /// This is a constant or a draw from the Wishart factor.
    sigma_inv : DMatrix<f64>,

    //sigma_qr : QR<f64, Dynamic, Dynamic>,
    //sigma_lu : LU<f64, Dynamic, Dynamic>,
    //sigma_det : f64,
    //prec : DMatrix<f64>,
    //eta : (DVector<f64>, DMatrix<f64>),
    // prec : DMatrix<f64>,

    /// By setting a shift and scale, sampling and log-prob is not done with respect
    /// to Self, anymore, but to the related multinormal:
    /// N(scale * self + shift; scale * shift * scale^T);
    op : Option<LinearOp>,

    loc_factor : Option<Box<MultiNormal>>,

    // Mixture distributions break the 1:1 mapping between rust ownership DAG and the
    // probabilistic graph, which in this case has a divergent structure from a common discrete factor.
    // We represent mixtures as a "default" value for the current distribution (which also holds
    // the k-dimensional data matrix and the discrete factor); and the other continuous sister factors
    // as "appended" to this one. When sampling and calculating log-probabilities, if the discrete
    // factor yields its default value, we sample from this distribution as usual; if not, we dispatch
    // the call to one of the "sister" mixing factors held in this field.
    mix : Option<Vec<Box<MultiNormal>>>,

    // Mixture is treated as a factor instead of a likelihood; because what alternates
    // is the mean parameter of the likelihood, not the likelihood itself.
    // mixture : Option<Mixture>,

    scale_factor : Option<Wishart>,

    /// This is a single-element vector. Unlike the univariate distributions (Poisson, Bernoulli)
    /// that hold conditional expectations and thus have a log-partition with the same size as eta,
    /// the multinormal holds a single parameter value, with its corresponding scalar log-partition.
    log_part : DVector<f64>,

    traj : Option<Trajectory>,

    obs : Option<DMatrix<f64>>,

    fixed_obs : Option<DMatrix<f64>>,
    
    names : Vec<String>,
    
    fixed_names : Option<Vec<String>>,
    
    fixed : bool
    
    // Vector of scale parameters; against which the Wishart factor
    // can be updated. Will be none if there is no scale factor.
    // suff_scale : Option<DVector<f64>>
}

/// Results from MultiNormal::conditional. The output will depend on how many
/// fixed and observed variables you have: For a single univariate observed variable,
/// you will get a Univariate variant; For more than one observed variable, you will
/// get a multivariate variant. 
pub enum ConditionalNormal {
    Univariate(Normal),
    Multivariate(MultiNormal)
}

impl MultiNormal {

    /*/// Returns the distribution with the parameters set to its
    /// gaussian approximation (mean and standard error).
    fn mle<'a>(y : impl Iterator<Item='a f64>) -> Result<Self, anyhow::Error> {
        let mut all =
        let n = y.nrows() as f64;
        if y.nrows() < y.ncols() {
            return Err(anyhow::Error::msg("MLE cannot be calculated if n is smaller than the number of parameters"));
        }
        let suf = Self::sufficient_stat(y);
        let mut mu : DVector<f64> = suf.column(0).clone_owned();
        mu.unscale_mut(n);
        let mut sigma = suf.remove_column(0);
        sigma.unscale_mut(n);
        Self::new(y.nrows(), mu, sigma)
    }*/

    pub fn mle(y : DMatrixSlice<'_, f64>) -> (DVector<f64>, DMatrix<f64>) {
        /*let n = y.nrows() as f64;
        let suf = Self::sufficient_stat(y);

        let mut mu_mle : DVector<f64> = suf.column(0).clone_owned();
        mu_mle.unscale_mut(n);

        let mut sigma_mle : DMatrix<f64> = suf.columns(1, suf.ncols() - 1).clone_owned();
        sigma_mle.unscale_mut(n);

        let mu_cross = mu_mle.clone() * mu_mle.transpose();
        sigma_mle -= mu_cross;

        (mu_mle, sigma_mle)*/

        let mu = y.row_mean().transpose();
        let mut err = y.clone_owned();
        for mut row in err.row_iter_mut() {
            row.iter_mut().enumerate().for_each(|(i, mut y)| *y = *y - mu[i] );
        }

        let cov = err.clone().transpose() * err;

        println!("{} {}", mu, cov);

        (mu, cov)
    }

    pub fn is_fixed(&self) -> bool {
        self.fixed_obs.is_some()
    }

    pub fn fixed_observations(&self) -> Option<&DMatrix<f64>> {
        self.fixed_obs.as_ref()
    }

    /*pub fn scaled<const N : usize>(cov : impl AsRef<[[f64; N]]>) {
        let (nrow, ncol) = (cov.len(), N);
        if nrow != self.mu.nrows() {
            panic!("Invalid multinormal dimension");
        }
        self.sigma = DMatrix::from_iterator(nrow, ncol, cov.iter().flatten());
    }*/

    pub fn invert_scale(s : &DMatrix<f64>) -> DMatrix<f64> {
        let s_qr = QR::<f64, Dynamic, Dynamic>::new(s.clone());
        // println!("s = {}", s);
        s_qr.try_inverse().unwrap()
        //
        // self.prec = self.
    }

    pub fn corr_from(mut cov : DMatrix<f64>) -> DMatrix<f64> {
        assert!(cov.nrows() == cov.ncols());
        let mut diag_m = DMatrix::zeros(cov.nrows(), cov.ncols());
        let diag = cov.diagonal().map(|d| 1. / d.sqrt() );
        diag_m.set_diagonal(&diag);
        cov *= &diag_m;
        diag_m *= cov;
        diag_m
    }

    /// Creates a centered multinormal with identity covariance of size n.
    pub fn new_standard(n : usize, p : usize) -> Self {
        let mu = DVector::zeros(p);
        let mut cov = DMatrix::zeros(p, p);
        cov.fill_with_identity();
        Self::new(n, mu, cov).unwrap()
    }

    /// Creates a non-centered multinormal with specified diagonal covariance.
    pub fn new_isotropic(n : usize, mu : impl AsRef<[f64]>, var : f64) -> Self {
        let n = mu.as_ref().len();
        let mut cov = DMatrix::zeros(n, n);
        cov.set_diagonal(&DVector::from_element(n, var));
        let v = Vec::from_iter(mu.as_ref().iter().cloned());
        Self::new(n, DVector::from(v), cov).unwrap()
    }

    /// Builds a new multivariate distribution from a mu vector and positive-definite
    /// covariance matrix sigma.
    pub fn new(n : usize, mu : DVector<f64>, sigma : DMatrix<f64>) -> Result<Self, anyhow::Error> {
        // let suff_scale = None;
        let log_part = DVector::from_element(1, 0.0);
        if !is_pd(sigma.clone()) {
            return Err(anyhow::Error::msg("Informed matrix is not positive-definite").into());
        }
        if mu.nrows() != sigma.nrows() || mu.nrows() != sigma.ncols() {
            return Err(anyhow::Error::msg("Mismatch between mean vector and sigma covariance sizes"));
        }
        //println!("sigma = {}", sigma);
        let sigma_inv = QR::new(sigma.clone()).try_inverse()
            .ok_or(anyhow::Error::msg("Informed matrix is not invertible"))?;
        // let func = CovFunction::None;
        // let lin_sigma_inv = LinearCov::None;
        /*let mu = param.0;
        let mut sigma = param.1;
        let eta = Self::eta( &(mu.clone(), sigma.clone()) );
        let sigma_qr = QR::<f64, Dynamic, Dynamic>::new(sigma.clone());
        let sigma_lu = LU::new(sigma.clone());
        let sigma_det = sigma_lu.determinant();
        let prec = sigma_qr.try_inverse().unwrap();*/
        // Self { mu, sigma, sigma_qr, sigma_lu, sigma_det, prec, eta }
        let mut norm = Self {
            n,
            mu : mu.clone(),
            sigma : sigma.clone(),
            sigma_inv,
            loc_factor: None,
            scale_factor : None,
            op : None,
            log_part,
            traj : None,
            scaled_mu : mu.clone(),
            obs : None,
            names : Vec::new(),
            fixed_names : None,
            fixed : false,
            fixed_obs : None,
            mix : None
            // mixture : None
        };
        norm.set_parameter(mu.rows(0, mu.nrows()), true);
        norm.set_cov(sigma.clone());
        Ok(norm)
    }

    pub fn get_mean(&self, ix : usize) -> f64 {
        self.mu[ix]
    }

    pub fn get_variance(&self, ix : usize) -> f64 {
        self.sigma[(ix, ix)]
    }

    pub fn get_stddev(&self, ix : usize) -> f64 {
        self.get_variance(ix).sqrt()
    }

    /// Marginal correlation between a pair of variables (ignoring any interactions for k >= 3 variables)
    pub fn marginal_correlation(&self, a : usize, b : usize) -> f64 {
        self.sigma[(a, b)] / (self.sigma[(a, a)] * self.sigma[(b, b)]).sqrt()
    }

    /// Partial correlation between a pair of variables (the residual correlations between
    /// the residuals of the two variables, when each variable is regressed against all other
    /// variables). This accounts for interactions among k >= 3 variables.
    pub fn partial_correlation(&self, a : usize, b : usize) -> f64 {
        self.sigma_inv[(a, b)] / (self.sigma_inv[(a, a)] * self.sigma_inv[(b, b)]).sqrt()
    }

    /// Scales this multinormal by the informed value.
    pub fn scale_by(&mut self, scale : DMatrix<f64>) {
        match self.op.take() {
            Some(mut op) => {
                assert!(op.scale.nrows() == scale.nrows());
                assert!(op.scale.ncols() == scale.ncols());
                op.update_scale(&self, scale);
                self.op = Some(op);
            },
            None => {
                self.op = Some(LinearOp::new(&self, None, Some(scale)));
            }
        }
    }

    /// Shifts this multinormal by the informed value.
    pub fn shift_by(&mut self, shift : DVector<f64>) {
        match self.op.take() {
            Some(mut op) => {
                op.update_shift(&self, shift);
                self.op = Some(op);
            },
            None => {
                self.op = Some(LinearOp::new(&self, Some(shift), None));
            }
        }
    }

    pub fn sigma_determinant<S>(cov : &Matrix<f64, Dynamic, Dynamic, S>) -> f64
    where
        S : Storage<f64, Dynamic, Dynamic>
    {
        let sigma_lu = LU::new(cov.clone_owned());
        let sigma_det = sigma_lu.determinant();
        sigma_det
    }

    fn multinormal_log_part(
        mu : DVectorSlice<'_, f64>,
        scaled_mu : DVectorSlice<'_, f64>,
        cov : &DMatrix<f64>,
        prec : &DMatrix<f64>
    ) -> f64 {
        let sigma_det = Self::sigma_determinant(&cov);

        // let prec_lu = LU::new(prec.clone());
        // let prec_det = prec_lu.determinant();
        // let p_mu_cov = -0.25 * scaled_mu.clone().transpose() * cov * &scaled_mu;

        //println!("sigma det ln: {}", sigma_det.ln());
        -0.5 * (mu.clone().transpose() * scaled_mu)[0] - 0.5 * sigma_det.ln()
        // p_mu_cov[0] - 0.5*sigma_det.ln()
    }

    /// Returns the reduced multivariate normal [ix, ix+n) by marginalizing over
    /// the remaining indices [0, ix) and [ix+n,p). This requires a simple copy
    /// of the corresponding entries. The scale and shift matrices are also
    /// sliced at the respective entries. TODO return ns : NormalSlice, then ns.clone_multivariate()
    pub fn marginal(&self, ix : usize, n : usize) -> MultiNormal {
        // let size = n - ix;
        // let mut marg = self.clone();
        let mu = DVector::from(Vec::from_iter(self.mu.rows(ix, n).iter().cloned()));
        let sigma = self.sigma.slice((ix, ix), (n, n)).clone_owned();
        let mut mn = MultiNormal::new(1, mu, sigma).unwrap();

        // if let Some(obs) = self.obs {
        //    let obs_t = obs.clone().transpose();
        //    mn.observe(obs_t.column_iter().map(|c| c[ix..ix+n].as_slice() ));
        // }

        // Modify mu
        // Modify sigma
        // Modify scale factors

        mn
    }

    // Splits this distribution into n independent others, assuming
    // pub fn split_independent(mut self, n : usize) -> Vec<MultiNormal>

    /*pub fn join_independent(ds : impl Iterator<Item=MultiNormal>) -> Self {
        unimplemented!()
    }*/

    /*pub(crate) fn marginal_mu_mut(&mut self, pos : usize, n : usize) -> MatrixSliceMut<'_, f64, Dynamic, U1, U1, Dynamic> {
        self.mu.rows_mut(pos, n)
    }

    /// Assuming independence, get the marginal diagonal block.
    pub(crate) fn marginal_sigma_block_diag_mut(&mut self, pos : usize, n : usize) -> DMatrixSliceMut<'_, f64> {
        self.sigma.slice_mut((pos, pos), (n, n))
    }*/

    /// Returns the reduced multivariate normal [0, n) by conditioning
    /// over entries [n,p) at the informed value. This operation condition
    /// on a user-supplied constant and has nothing to do with the
    /// Conditional<Distribution> trait, which is used for conditioning over
    /// other random variables.
    /// conditioning a multivariate normal by fixing a subset of its variables is implemented via
    /// a least-squares problem.
    /// TODO return ns : NormalSlice, then clone_multivariate OR clone_univariate.
    pub fn conditional(&self, value : DVector<f64>) -> ConditionalNormal {
        /*let mut cond = self.marginal(0, value.nrows());
        cond.cond_update_from(&self, value);
        cond*/
        unimplemented!()
    }

    fn update_cond_mean(
        &mut self,
        partial_sigma_inv : &DMatrix<f64>,
        upper_cross_cov : &DMatrixSlice<'_,f64>,
        marg_mu : DVectorSlice<'_, f64>,
        fix_mu : DVectorSlice<'_, f64>,
        mut value : DVector<f64>,
    ) {
        value.sub_assign(&fix_mu);
        let mu_off_scaled = partial_sigma_inv.clone() * value;
        let final_mu_off = upper_cross_cov * mu_off_scaled;
        self.mu.copy_from(&marg_mu);
        self.mu.sub_assign(&final_mu_off);
    }

    // pub(crate) fn partial_cond_log_prob(&self, pos : usize, n : usize) -> Option<DVector<f64>> {
    // }

    /// Evaluate the conditional log-probability of all samples, assuming the distribution
    /// has no factors.
    pub fn cond_log_prob(&self) -> Option<DVector<f64>> {
        assert!(self.loc_factor.is_none());
        assert!(self.op.is_none());
        let ys = self.obs.as_ref()?;

        let sigma_det = Self::sigma_determinant(&self.sigma);
        let k = self.mu.nrows() as f64;
        let base_measure = ((2. * PI).powf(-k / 2.) * sigma_det.powf(-0.5)).ln();

        let mut lp = DVector::zeros(ys.nrows());
        for (mut lp, y) in lp.iter_mut().zip(ys.row_iter()) {
            let maha = self.mahalanobis(y.iter());
            println!("maha = {}", maha);
            *lp = base_measure - maha / 2.;
        }
        Some(lp)
    }

    fn mahalanobis<'a>(&self, y : impl Iterator<Item=&'a f64>) -> f64 {
        let y = DVector::from(y.cloned().collect::<Vec<_>>());
        let err = y - &self.mu;
        let err_t = err.clone().transpose();
        (err_t * &self.sigma_inv * &err)[0]
    }

    fn update_cond_cov(
        &mut self,
        partial_sigma_inv : &DMatrix<f64>,
        marg_sigma : DMatrixSlice<'_, f64>,
        upper_cross_cov : DMatrixSlice<'_,f64>,
        lower_cross_cov : DMatrixSlice<'_,f64>
    ) {
        let mut sigma_off = upper_cross_cov.clone_owned();
        sigma_off *= partial_sigma_inv;
        sigma_off *= lower_cross_cov;
        //self.sigma.copy_from(&marg_sigma);
        let mut sigma : DMatrix<f64> = marg_sigma.clone_owned();
        sigma.sub_assign(&sigma_off);
        self.sigma_inv = Self::invert_scale(&sigma);
    }

    /// Updates the internal state of self to reflect the conditional distribution
    /// of the joint parameter when the values joint[value.len()..n]
    /// are held fixed at the informed value.
    pub fn cond_update_from(&mut self, joint : &MultiNormal, mut value : DVector<f64>) {
        let fix_n = value.nrows();
        let cond_n = joint.mu.nrows() - fix_n;
        let marg_mu = joint.mu.rows(0, cond_n);
        let sigma = Self::invert_scale(&joint.sigma_inv);
        let marg_sigma = sigma.slice((0, 0), (cond_n, cond_n));
        let partial_sigma : DMatrix<f64> = sigma
            .slice((cond_n, cond_n), (fix_n, fix_n))
            .clone_owned();
        let partial_sigma_inv = Self::invert_scale(&partial_sigma);
        let upper_cross_cov = sigma
            .slice((0, cond_n), (cond_n, fix_n));
        let lower_cross_cov = sigma
            .slice((cond_n, 0), (fix_n, cond_n));
        let fix_mu = joint.mu.rows(cond_n, fix_n);
        self.update_cond_mean(&partial_sigma_inv, &upper_cross_cov, marg_mu, fix_mu, value);
        self.update_cond_cov(&partial_sigma_inv, marg_sigma, upper_cross_cov, lower_cross_cov);
        self.op = None;
        self.scaled_mu = self.sigma_inv.clone() * &self.mu;
    }

    pub fn linear_mle(x : &DMatrix<f64>, y : &DVector<f64>) -> Self {
        assert!(y.nrows() == x.nrows());
        let n = x.nrows() as f64;
        let ols = fit::linear::OLS::estimate(&y, &x);
        let mu = ols.beta;
        let cov = (x.clone().transpose() * x).scale(ols.err.unwrap().sum() / n);
        Self::new(y.nrows(), mu, cov).unwrap()
    }

    pub fn set_cov(&mut self, cov : DMatrix<f64>) {
        // assert!(crate::distr::is_pd(cov.clone()));
        if self.sigma.nrows() == cov.nrows() && self.sigma.ncols() == cov.ncols() {
            self.sigma.copy_from(&cov);
        } else {
            if cov.nrows() == cov.ncols() && cov.nrows() == self.mu.nrows() {
                self.sigma = cov;
            } else {
                panic!(
                    "Mismatch between mean vector and covariance matrix dimensions ({} vs. ({} x {}))",
                    self.mu.nrows(),
                    cov.nrows(),
                    cov.ncols()
                );
            }
        }
        let prec = Self::invert_scale(&self.sigma);
        self.scaled_mu = prec.clone() * &self.mu;
        self.sigma_inv = prec;
        let mu = self.mu.clone();
        self.update_log_partition();
        if let Some(mut op) = self.op.take() {
            op.update_from(&self);
            self.op = Some(op);
        }
    }
    
    /*/// Watches for the informed variable names at any sample implementors, 
    /// interpreting the variable values as fixed. For example, is this distribution is 
    /// a multinormal of dimension p, after fixing for k informed variables, the distribution
    /// will actually be a (p-k)-variate reduced joint distribution, which is a linear function of
    /// the fixed k's and the current parameter vector. The basis for building a regression model
    /// is creating an observed univariate node, and then creating a completely fixed (pass k names
    /// for a k dimensional distribution) multinormal distribution. Alternatively, you can fix p-1
    /// factors by passing p-1 names to fix and 1 name to observe (which will leave a single random obsserved node) 
    /// and then call Self::conditional,
    /// which will return a Multi or Univariate node with the remaining fixed nodes as factors.
    /// Informing n will fix the n last variables.
    fn fix(mut self, n : usize) -> MultiNormal {
        unimplemented!()
    }
    
    /// Fix all elements but the first one, returning a univariate node which has the
    /// fixed nodes as its mean parameter.
    fn fix_remaining(mut self) -> Normal {
        unimplemented!()
    }
    
    /// Fix all values.
    fn fix_all(mut self) -> Normal {
        unimplemented!()
    }*/
    
    /*pub fn new(mu : DVector<f64>, w : Wishart) -> Self {
        Self { mu : mu, cov : Covariance::new_random(w), shift : None, scale : None }
    }

    pub fn new_standard(n : usize) -> Self {
        let mu = DVector::from_element(n, 0.0);
        let mut sigma = DMatrix::from_element(n, n, 0.0);
        sigma.set_diagonal(&DVector::from_element(n, 1.));
        Self::new_scaled(mu, sigma)
    }

    fn dim(&self) -> usize {
        self.mu.shape().0
    }

    fn n_params(&self) -> usize {
        self.mu.shape().0 + self.sigma.shape().0 * self.sigma.shape().1
    }*/
    
    /*/// Creates this distribution as a fixed factor. Sampling from this distribution
    /// now means sampling the regression coefficient vector (which equals the 
    /// Sigma_12 Sigma_22^-1 matrix), where the Sigma_11 means the variance (or covariance)
    /// of the child node, and Sigma_22 means the covariance of self. A multinormal in fixed mode
    /// carries data which is not assumed to have been sampled from it: Samples of Self just means
    /// what coefficients to use in the conditional multinormal expression to build the
    /// expected value for the child factor. If used as a factor in a conditional expectation
    /// expression and fixed is not called, any sample(.) and log_prob(.) calculations over the
    /// child node must fail. The fixed switch alters how sample(.) behaves: if it is off, the
    /// actual sample of dimension p is emitted; if it is on, sample will return the nx1 vector
    /// resulting form the matrix multiplication x*b where b is the underlying multinormal
    /// random sample. Moreover, a fixed multinormal does not have an covariance matrix that is
    /// independent of the conditioned factor: It is a function of this factor, and cannot be
    /// altered by the outside world independent of it.
    pub fn fixed(&mut self) -> &mut Self {
        self.fixed = true;
        self
    }*/

    /*pub fn fixed_from_sample(sample : &dyn Sample, coefs : &[f64], names : &[&str]) -> Option<Self> {
        let obs = collect_from_sample(sample, names)?;
        let mut mn = MultiNormal::new_standard(obs.nrows(), obs.ncols());
        mn.mu.iter_mut().enumerate().for_each(|(ix, m)| *m = coefs[ix] );
        mn.fixed_obs = Some(obs);
        Some(mn)
    }*/

    /*pub fn from_sample(sample : &dyn Sample, names : &[&str]) -> Option<Self> {
        let obs = collect_from_sample(sample, names)?;
        let mut mn = MultiNormal::new_standard(obs.nrows(), obs.ncols());
        mn.obs = Some(obs);
        Some(mn)
    }*/

}

fn collect_from_sample(sample : &dyn Sample, names : &[&str]) -> Option<DMatrix<f64>> {
    let mut joint_vars = Vec::new();
    let mut n = 0;
    for name in names.iter() {
        match sample.variable(name) {
            Variable::Real(r) => {
                joint_vars.extend(r);
                if n == 0 {
                    n = joint_vars.len();
                }
            },
            _ => return None
        }
    }
    assert!(n > 0);
    let p = joint_vars.len() / n;
    Some(DMatrix::from_vec(n, p, joint_vars))
}

impl ExponentialFamily<Dynamic> for MultiNormal {

    // This is the **summed** base measure, left outside the log-probability summation.
    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        DVector::from_element(1, (2. * PI).powf( - (y.ncols() as f64) / 2. ) )
    }

    /// Returs the matrix [sum(yi) | sum(yi @ yi^T)]
    /// TODO if there is a constant scale factor, the sufficient statistic
    /// is the sample row-sum. If there is a random scale factor (wishart) the sufficient
    /// statistic is the matrix [rowsum(x)^T sum(x x^T)], which is guaranteed to be
    /// positive-definite.
    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        /*let r_sum = y.row_sum();
        let cross_p = y.clone_owned().transpose() * &y;
        let mut t = cross_p.add_column(0);
        t.column_mut(0).copy_from(&r_sum);
        t*/
        let yt = y.clone_owned().transpose();
        let mut t = DMatrix::zeros(y.ncols(), y.ncols() + 1);
        let mut ss_ws = DMatrix::zeros(y.ncols(), y.ncols());
        for (yr, yc) in y.row_iter().zip(yt.column_iter()) {
            t.slice_mut((0, 0), (t.nrows(), 1)).add_assign(&yc);
            yc.mul_to(&yr, &mut ss_ws);
            t.slice_mut((0, 1), (t.ncols() - 1, t.ncols() - 1)).add_assign(&ss_ws);
        }
        t
    }

    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {

        // The suff log prob calculation is reserved for multinormals in the position
        // of priors or that have a single observation data point. Use cond_log_prob otherwise.
        assert!(self.obs.as_ref().map(|obs| obs.nrows() == 1).unwrap_or(true));

        if let Some(op) = &self.op {
            return op.dst.suf_log_prob(t);
        }
        let mut lp = 0.0;
        println!("suff stat = {:?}", t.shape());
        println!("scaled mu = {:?}", self.scaled_mu.shape());

        let (sum, sum_sq) = (t.column(0), t.columns(1, t.ncols() - 1));
        lp += self.scaled_mu.dot(&sum);
        for (s_inv_row, sq_row) in self.sigma_inv.row_iter().zip(sum_sq.row_iter()) {
            lp += (-0.5) * s_inv_row.dot(&sq_row);
        }
        let log_part = self.log_partition();

        // lp + self.obs.as_ref().unwrap().nrows() as f64 * log_part[0]

        lp + log_part[0]
    }

    fn log_partition<'a>(&'a self) -> &'a DVector<f64> {
        &self.log_part
    }

    /// Updating the log-partition of a multivariate normal assumes a new
    /// mu vector (eta); but using the currently set covariance value.
    /// The log-partition of the multivariate normal is a quadratic form of
    /// the covariance matrix (taking the new location as parameter) that
    /// has half the log of the determinant of the covariance (scaled by -2) subtracted from it.
    fn update_log_partition<'a>(&'a mut self, /*eta : DVectorSlice<'_, f64>*/ ) {
        // TODO update eta parameter here.
        // let cov = Self::invert_scale(&self.sigma_inv);
        let log_part = Self::multinormal_log_part(
            self.mu.rows(0, self.mu.nrows()),
            self.scaled_mu.rows(0, self.scaled_mu.nrows()),
            &self.sigma,
            &self.sigma_inv
        );
        self.log_part = DVector::from_element(1, log_part);
        //let sigma_lu = LU::new(cov.clone());
        //let sigma_det = sigma_lu.determinant();
        //let p_eta_cov = -0.25 * eta.clone().transpose() * cov * &eta;
        //self.log_part = DVector::from_element(1, p_eta_cov[0] - 0.5*sigma_det.ln())
    }

    // eta = [sigma_inv*mu,-0.5*sigma_inv] -> [mu|sigma]
    // N(mu|sigma) has link sigma^(-1/2) mu (the exponential form
    // parametrized by a natural parameter eta requires the normal
    // conditional on a realized scale factor).
    fn link_inverse<S>(_eta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        /*let theta_1 = -0.5 * eta.1.clone() * eta.0.clone();
        let theta_2 = -0.5 * eta.1.clone();
        (theta_1, theta_2)*/
        unimplemented!()
    }

    // theta = [mu|sigma] -> [sigma_inv*mu,-0.5*sigma_inv]
    fn link<S>(_theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        /*let qr = QR::<f64, Dynamic, Dynamic>::new(theta.1.clone());
        let eta_1 = qr.solve(&theta.0).unwrap();
        let eta_2 = -0.5 * qr.try_inverse().unwrap();
        let mut eta = eta_2.add_column(0);
        eta.column_mut(0).copy_from(&eta_1);
        eta*/
        unimplemented!()
    }

    // Remember that data should be passed as wide rows here.
    // We have that dbeta/dy = sigma_inv*(y - mu)
    // If the multinormal is linearized, we apply the linear operator
    // to the derivative: x^T sigma_inv*(y - mu).
    fn grad(&self, y : DMatrixSlice<'_, f64>, x : Option<DMatrix<f64>>) -> DVector<f64> {
        /*let cov_inv = self.cov_inv().unwrap();
        assert!(cov_inv.nrows() == cov_inv.ncols());
        assert!(cov_inv.nrows() == self.mean().nrows());

        // Remember that y comes as wide rows - We make them columns here.
        let yt = y.transpose();
        let yt_scaled = cov_inv.clone() * yt;
        // same as self.scaled_mu - But we need to reallocate here
        let m_scaled = cov_inv * self.mean();
        let ys = yt_scaled.column_sum(); // or mean?
        let mut score = yt_scaled - m_scaled;
        if let Some(op) = &self.op {
            score = op.scale.transpose() * score.clone();
        }
        score*/

        // Implements IRLS step
        let n = y.nrows();
        if let Some(x) = self.fixed_obs.as_ref() {
            unimplemented!()
        } else {
            panic!("Missing fixed observatios to calculate coefficients");
        }
    }

    /*fn update_grad(&mut self, _eta : DVectorSlice<'_, f64>) {
        unimplemented!()
    }

    fn grad(&self) -> &DVector<f64> {
        unimplemented!()
    }*/

}

#[test]
fn suff_stat() {
    let x = DMatrix::from_row_slice(3,3,&[
        1.0, 1.0, 1.0,
        2.0, 2.0, 2.0,
        3.0, 3.0, 3.0
    ]);
    println!("norm suff = {}", MultiNormal::sufficient_stat(x.slice((0, 0), (3,3))));
}

impl Distribution for MultiNormal {

    fn view_parameter(&self, natural : bool) -> &DVector<f64> {
        match natural {
            true => &self.scaled_mu,
            false => &self.mu
        }
    }

    fn dyn_factors(&self) -> (Option<&dyn Distribution>, Option<&dyn Distribution>) {
        let loc = self.loc_factor.as_ref().map(|lf| lf.as_ref() as &dyn Distribution);
        let scale = self.scale_factor.as_ref().map(|sf| sf as &dyn Distribution);
        (loc, scale)
    }

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Distribution>, Option<&mut dyn Distribution>) {
        let loc = self.loc_factor.as_mut().map(|lf| lf.as_mut() as &mut dyn Distribution);
        let scale = self.scale_factor.as_mut().map(|sf| sf as &mut dyn Distribution);
        (loc, scale)
    }

    fn set_natural<'a>(&'a mut self, eta : &'a mut dyn Iterator<Item=&'a f64>) {
        self.mu = DVector::from(Vec::from_iter(eta.cloned()));

        // Use iterator to avoid allocation
        // self.mu.iter_mut().zip(eta).for_each(|(old, new)| *old = *new );

        self.scaled_mu = self.sigma_inv.clone() * &self.mu;
        self.update_log_partition( /*p*/ );

        if let Some(mut op) = self.op.take() {
            op.update_from(&self);
            self.op = Some(op);
        }
    }

    /// Set parameter should verify if there is a scale factor. If there is,
    /// also sets the eta of those factors and write the new implied covariance
    /// into self. Only after the new covariance is set to self, call self.update_log_partition().
    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, natural : bool) {
        assert!(natural);
        self.set_natural(&mut p.iter());
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        match &self.op {
            Some(op) => op.dst.mean(),
            None => &self.mu
        }
    }

    fn mode(&self) -> DVector<f64> {
        self.mean().clone()
    }

    fn var(&self) -> DVector<f64> {
        match &self.op {
            Some(op) => op.dst.cov().unwrap().diagonal(),
            None => self.cov().unwrap().diagonal()
        }
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        match &self.op {
            Some(op) => Some(op.dst.sigma.clone()),
            None => Some(self.sigma.clone())
        }
    }

    fn cov_inv(&self) -> Option<DMatrix<f64>> {
        match &self.op {
            Some(op) => Some(op.dst.sigma_inv.clone()),
            None => Some(self.sigma_inv.clone())
        }
    }

    fn joint_log_prob(&self, /*y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>*/ ) -> Option<f64> {
        if let Some(op) = &self.op {
            return op.dst.joint_log_prob();
        }
        let y = self.obs.as_ref()?;
        let t = Self::sufficient_stat(y.slice((0, 0), (y.nrows(), y.ncols())));
        let lp = self.suf_log_prob(t.slice((0, 0), (t.nrows(), t.ncols())));
        let loc_lp = match &self.loc_factor {
            Some(loc) => {
                let mu_row : DMatrix<f64> = DMatrix::from_row_slice(
                    self.mu.nrows(),
                    1,
                    self.mu.data.as_slice()
                );
                loc.suf_log_prob(mu_row.slice((0, 0), (0, self.mu.nrows())))
            },
            None => 0.0
        };
        let scale_lp = match &self.scale_factor {
            Some(scale) => {
                let sinv_diag : DVector<f64> = self.sigma_inv.diagonal().clone_owned();
                scale.suf_log_prob(sinv_diag.slice((0, 0), (sinv_diag.nrows(), 1)))
            },
            None => 0.0
        };
        Some(lp + loc_lp + scale_lp)
    }

    fn sample_into(&self, mut dst : DMatrixSliceMut<'_, f64>) {
        use rand::prelude::*;
        
        if dst.nrows() != self.n {
            let dst_dim = dst.shape();
            let distr_dim = (self.n, self.mu.nrows());
            panic!("Error (sample_into): destination has dimension {:?} but distribution requires {:?}", dst_dim, distr_dim);
        }
        
        // Populate destination with independent standard normal draws
        for i in 0..self.n {
            for j in 0..self.mu.nrows() {
                dst[(i, j)] = rand::thread_rng().sample(rand_distr::StandardNormal);
            }
        }
        
        // TODO add field cached_cov_l : Option<RefCell<DMatrix<f64>>> to avoid this computation.
        let sigma_lu = LU::new(self.sigma.clone());
        let sigma_low = sigma_lu.l();
        
        let mut row_t = DVector::zeros(self.mu.nrows());
        let mut scaled_row = DVector::zeros(self.mu.nrows());
        
        for mut row in dst.row_iter_mut() {
            row.transpose_to(&mut row_t);
            
            // Offset samples by mu
            row_t += &self.mu;
            
            // Scale samples by lower Cholesky factor of covariance matrix (matrix square root)
            sigma_low.mul_to(&row_t, &mut scaled_row);
            row.copy_from_slice(scaled_row.as_slice());
        }
    }

    //fn observations(&self) -> Option<&DMatrix<f64>> {
    //    self.obs.as_ref()
    //}

    /*fn set_observations(&mut self, obs : &DMatrix<f64>) {
        assert!(obs.ncols() == self.mu.nrows());
        if let Some(ref mut old_obs) = self.obs {
            old_obs.copy_from(&obs);
        } else {
            self.obs = Some(obs.clone());
        }
    }*/

}

impl Markov for MultiNormal {

    fn natural_mut<'a>(&'a mut self) -> DVectorSliceMut<'a, f64> {
        self.mu.column_mut(0)
    }

    fn canonical_mut<'a>(&'a mut self) -> Option<DVectorSliceMut<'a, f64>> {
        None
    }

}

impl Posterior for MultiNormal {

    /*fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Posterior>, Option<&mut dyn Posterior>) {
        let loc = self.loc_factor.as_mut().map(|lf| lf.as_mut() as &mut dyn Posterior);
        let scale = self.scale_factor.as_mut().map(|sf| sf as &mut dyn Posterior);
        (loc, scale)
    }*/

    fn approximation_mut(&mut self) -> Option<&mut MultiNormal> {
        Some(self)
    }

    fn approximation(&self) -> Option<&MultiNormal> {
        Some(self)
    }

    fn trajectory(&self) -> Option<&Trajectory> {
        self.traj.as_ref()
    }

    fn trajectory_mut(&mut self) -> Option<&mut Trajectory> {
        self.traj.as_mut()
    }
    
    fn start_trajectory(&mut self, size : usize) {
        self.traj = Some(Trajectory::new(size, self.view_parameter(true).nrows()));
    }
    
    /// Finish the trajectory before its predicted end.
    fn finish_trajectory(&mut self) {
        self.traj.as_mut().unwrap().closed = true;
    }

}

/*impl Prior for MultiNormal {

    type Parameter = &[f64];

    fn prior(param : &[f64]) -> {
        MultiNormal::new_isotropic(param, 1.0)
    }

}*/

impl Likelihood<[f64]> for MultiNormal {

    fn view_variables(&self) -> Option<Vec<String>> {
        Some(self.names.clone())
    }
    
    fn likelihood<'a>(obs : impl IntoIterator<Item=&'a [f64]>) -> Self {
        // let n : usize = obs.clone().count();
        // let p : usize = obs.next().unwrap().len();
        let mut mn = MultiNormal::new_standard(1, 1);
        mn.observe(obs);
        mn
    }

    fn observe<'a>(&mut self, obs : impl IntoIterator<Item=&'a [f64]>) {
        // let p = obs.next().unwrap().len();
        // if p != self.mu.nrows() {
        //    panic!("Multinormal has dimension {} but data has dimension {}", self.mu.nrows(), p);
        // }
        // observe_multivariate_generic(self, obs.into_iter());

        let obs = collect_observations(obs.into_iter());
        self.obs = Some(obs);

        let (mean, cov) = MultiNormal::mle(self.obs.as_ref().unwrap().into());

        // Do this step to guarantee dimensions are expanded correctly.
        self.mu = mean.clone();
        self.sigma = cov.clone();

        // Do this to calculate scaled_mu, sigma_inv, log_partition, etc.
        self.set_cov(cov);
        self.set_parameter((&mean).into(), true);

    }
    
    /*fn factors_mut<'a>(&'a mut self) -> (Option<&'a mut dyn Posterior>, Option<&'a mut dyn Posterior>) {
        self.dyn_factors_mut()
    }*/

    fn with_variables(&mut self, vars : &[&str]) -> &mut Self {
        self.names = vars.iter().map(|v| v.to_string()).collect();
        self
    }
    
    fn view_fixed(&self) -> Option<Vec<String>> {
        self.fixed_names.clone()
    }
    
    fn with_fixed(&mut self, fixed : &[&str]) -> &mut Self {
        self.fixed_names = Some(fixed.iter().map(|s| s.to_string()).collect());
        self
    }
    
    fn view_variable_values(&self) -> Option<&DMatrix<f64>> {
        self.obs.as_ref()
    }

    fn view_fixed_values(&self) -> Option<&DMatrix<f64>> {
        self.fixed_obs.as_ref()
    }

    fn observe_sample(&mut self, sample : &dyn Sample, vars : &[&str]) {
        // let mut obs = self.obs.take().unwrap_or(DMatrix::zeros(self.n, self.mu.len()));
        // if let 
        // let names = self.names.clone();
        super::observe_real_columns(vars, sample, &mut self.obs, self.n);
        /*if let Some(fixed_names) = &self.fixed_names {
            let fix_names = fixed_names.clone();
            super::observe_real_columns(&fix_names[..], sample, &mut self.fixed_obs, self.n);
        }*/
        // self.obs = Some(obs);
    }

    /*fn visit_factors<F>(&mut self, f : F) where F : Fn(&mut dyn Posterior) {
        if let Some(ref mut loc) = self.loc_factor {
            f(loc.as_mut());
            loc.visit_post_factors(&f as &dyn Fn(&mut dyn Posterior));
        }
        if let Some(ref mut scale) = self.scale_factor {
            f(scale);
            scale.visit_post_factors(&f as &dyn Fn(&mut dyn Posterior));
        }
    }*/
}

impl<'a> Estimator<'a, &'a MultiNormal> for MultiNormal {

    type Algorithm = ();

    type Error = &'static str;

    //fn predict<'a>(&'a self, cond : Option<&'a Sample/*<'a>*/>) -> Box<dyn Sample/*<'a>*/> {
    //    unimplemented!()
    //}
    
    /*fn take_posterior(self) -> Option<MultiNormal> {
        unimplemented!()
    }
    
    fn view_posterior<'a>(&'a self) -> Option<&'a MultiNormal> {
        unimplemented!()
    }*/
    
    fn fit(&'a mut self, algorithm : Option<()>) -> Result<&'a MultiNormal, &'static str> {
        // self.observe_sample(sample);
        let y = self.obs.clone().unwrap();
        let n = y.nrows() as f64;
        match (&mut self.loc_factor, &mut self.scale_factor) {
            (Some(ref mut norm_prior), Some(ref mut gamma_prior)) => {
                unimplemented!()
            },
            (Some(ref mut norm_prior), None) => {
                // Calculate sample centroid
                let (mu_mle, _) = MultiNormal::mle(self.obs.as_ref().unwrap().into());

                // Scale centroid by precision of self
                let scaled_mu_mle = (self.sigma_inv.clone() * mu_mle).scale(n);

                // sum the weighted (precision-scaled) prior centroid with scaled sample centroid
                let w_sum_prec = norm_prior.sigma_inv.clone() + self.sigma_inv.scale(n);
                let w_sum_cov = Self::invert_scale(&w_sum_prec);
                norm_prior.mu = w_sum_cov * (norm_prior.scaled_mu.clone() + scaled_mu_mle);

                // substitute prior precision by scaled and summed precision
                norm_prior.sigma_inv = w_sum_prec;

                // substitute prior mean by scaled and summed mean
                norm_prior.scaled_mu = norm_prior.sigma_inv.clone() * &norm_prior.mu;
                Ok(&(*norm_prior))
            },
            _ => Err("Distribution does not have a conjugate location factor")
        }
        // Ok(&mut self.loc_factor)
    }

}

impl Observable for MultiNormal {

    fn observations(&mut self) -> &mut Option<DMatrix<f64>> {
        &mut self.obs
    }

    fn sample_size(&mut self) -> &mut usize {
        &mut self.n
    }

}

impl Display for MultiNormal {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MNorm({})", self.mu.nrows())
    }

}

pub mod utils {

    use nalgebra::*;
    //use num_traits::identities::Zero;

    /// Append a constant term to a predictor matrix. Modify to prepend_intercept.
    pub fn append_intercept<N : Scalar + From<f64>>(x : DMatrix<N>) -> DMatrix<N> {
        x.insert_column(0, N::from(1.0))
    }

    pub fn remove_intercept<N : Scalar + From<f64>>(x : DMatrix<N>) -> DMatrix<N> {
        x.remove_column(0)
    }

    /// Generates a matrix with a single intercept column.
    pub fn intercept_from_dim(dim : usize) -> DMatrix::<f64> {
        DMatrix::<f64>::from_element(dim, 1, 1.0)
    }

    /// Summed squared error
    pub fn sse(errors : DVector<f64>) -> f64 {
        errors.iter().fold(0.0, |err, e| (err + e).powf(2.0) )
    }

    pub fn append_or_intercept(x : Option<DMatrix<f64>>, nrows : usize) -> DMatrix<f64> {
        match x {
            Some(m) => append_intercept(m),
            None => intercept_from_dim(nrows)
        }
    }

    /*/// Given a multivariate parameter vector which actually refers to
    /// independent parameter columns of a d-variate random vector,
    /// pack it into a matrix
    pub fn pack_param_matrix<N>(b : &DVector<N>, d : usize) -> DMatrix<N>
    where
        N : Scalar + Zero
    {
        let p = b.nrows() / d;
        let mut b_mat = DMatrix::zeros(p, d);
        for j in 0..d {
            b_mat.column_mut(j).copy_from(&b.slice((j*d, 0), (p, 1)));
        }
        b_mat
    }*/

}

/// Converts to Normal if implementor has unique dimension.
impl TryInto<Normal> for MultiNormal {

    type Error = ();

    fn try_into(self) -> Result<Normal,()> {
        unimplemented!()
    }
}

/// Converts to a sequence of normals (respecting parameter order)
/// if covariance is diagonal.
impl TryInto<Vec<Normal>> for MultiNormal {

    type Error = ();

    fn try_into(self) -> Result<Vec<Normal>, ()> {
        unimplemented!()
    }

}

/// Creates MultiNormal of dimension 1 from univariate normal.
impl From<Normal> for MultiNormal {

    fn from(n : Normal) -> MultiNormal {
        unimplemented!()
    }

}

const EPS : f32 = 1E-8;

/// Verifies if the informed matrix is positive-definite (and can be used as a covariance matrix)
pub fn is_pd<N : Scalar + RealField + From<f32>>(m : DMatrix<N>) -> bool {
    let symm_m = build_symmetric(m.clone());
    if (m - &symm_m).sum() < N::from(EPS) {
        true
    } else {
        false
    }
}

/// Builds a symmetric matrix from M as (1/2)*(M + M^T)
pub fn build_symmetric<N : Scalar + RealField + From<f32>>(m : DMatrix<N>) -> DMatrix<N> {
    assert!(m.nrows() == m.ncols(), "approx_pd: Informed non-square matrix");
    let mt = m.transpose();
    (m + mt).scale(N::from(0.5))
}

/// Computes the spectral decomposition of square matrix m by applying the SVD to M M^T.
/// Returns the U matrix (left eigenvectors) and the eigenvalue diagonal D such that M = U D U^T
///
/// # References
/// Singular value decomposition
/// ([Wiki])(https://en.wikipedia.org/wiki/Singular_value_decomposition#Relation_to_eigenvalue_decomposition)
pub fn spectral_dec<N : Scalar + RealField + From<f32>>(m : DMatrix<N>) -> (DMatrix<N>, DVector<N>) {
    let mt = m.transpose();
    let m_mt = m * mt;
    let svd = linalg::SVD::new(m_mt, true, true);
    let eign_vals = svd.singular_values.clone();
    let u = svd.u.unwrap();
    (u, eign_vals.map(|e| e.sqrt()))
}

/// Transforms a potentially non-positive definite square matrix m into a positive
/// definite approximation (i.e. a matrix that defines a convex surface for the inner
/// product of any vector, and can be used as a covariance matrix). If m is PD already,
/// the output is no different that the input. This approximation minimizes the
/// [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm)
/// between the specified matrix and all the possible positive-definite matrices
///
/// # References
///
/// Higham, N. J. ([1988](https://www.sciencedirect.com/science/article/pii/0024379588902236)).
/// Computing a nearest symmetric positive semidefinite matrix.
/// Linear Algebra Appl., 103, 103–118. doi: 10.1016/0024-3795(88)90223-6 (Theorem 2.1).
pub fn approx_pd<N : Scalar + RealField + From<f32>>(m : DMatrix<N>) -> DMatrix<N> {
    let symm_m = build_symmetric(m);
    let (u, mut diag) = spectral_dec(symm_m);
    diag.iter_mut().for_each(|d| *d = if *d >= N::from(0.0) { *d } else { N::from(0.0) });
    let diag_m = DMatrix::from_diagonal(&diag);
    let u_t = u.transpose();
    u * diag_m * u_t
}

impl TryFrom<serde_json::Value> for MultiNormal {

    type Error = String;

    fn try_from(val : Value) -> Result<Self, String> {
        let mean_val = val.get("mean").ok_or("Missing 'mean' entry of MultiNormal node")?;
        let cov_val = val.get("cov").ok_or("Missing 'cov' entry of MultiNormal node")?;
        // let n = val.get("n").ok_or("Missing 'cov' entry of MultiNormal node")?;
        let mut mean = crate::model::parse_vector(mean_val)?;
        let mut cov = crate::model::parse_matrix(cov_val)?;
        Ok(MultiNormal::new(0, mean, cov).map_err(|e| format!("{}", e))?)
    }

}

impl TryFrom<Model> for MultiNormal {

    type Error = String;

    fn try_from(lik : Model) -> Result<Self, String> {
        match lik {
            Model::MN(m) => Ok(m),
            _ => Err(format!("Object does not have a top-level multinormal node"))
        }
    }

}

impl<'a> TryFrom<&'a Model> for &'a MultiNormal {

    type Error = String;

    fn try_from(lik : &'a Model) -> Result<Self, String> {
        match lik {
            Model::MN(m) => Ok(m),
            _ => Err(format!("Object does not have a top-level bernoulli node"))
        }
    }
}

fn collect_observations<'a>(values : impl Iterator<Item=&'a [f64]>) -> DMatrix<f64> {
    let mut p = 0;
    let mut m : Vec<f64> = Vec::new();
    values.for_each(|row| {
        if p == 0 {
            p = row.len();
        } else {
            assert!(p == row.len());
        }
        m.extend(row.iter());
    });
    let n = m.len() / p;
    let wide = DMatrix::from_vec(p, n, m);
    wide.transpose()
}

impl Fixed<[f64]> for MultiNormal {

    fn observe_fixed<'a>(&mut self, values : impl Iterator<Item=&'a [f64]>) {
        let fixed = collect_observations(values);
        self.fixed_obs = Some(fixed.transpose());
    }

    fn fixed<'a>(values : impl Iterator<Item=&'a [f64]>, coefs : impl Iterator<Item=&'a f64>) -> Self {
        let fixed = collect_observations(values);
        let mut mn = MultiNormal::new_standard(fixed.nrows(), fixed.ncols());
        let mu_vec : Vec<_> = coefs.cloned().collect();
        let mut mu = DVector::from_vec(mu_vec);
        assert!(mu.nrows() == fixed.ncols());
        mn.mu = mu;
        mn.fixed_obs = Some(fixed);
        mn
    }

    // fn observe_fixed<'a>(values : impl Iterator<Item=&'a Sample>, coefs : impl Iterator<Item=&'a f64>);
    // Updates an initially random distribution by fixig its values:
    // let m = MultiNormal::prior(&[0.2, 0.3]).fix(&[&[0.1, 0.2], &[1.2, 1.2]]);
    // fn fix(&mut self, obs : &[Sample]);
}

impl Mixture for MultiNormal {

    fn mixture<const K : usize>(distrs : [Self; K], probs : [f64; K]) -> Self {
        let mix : Vec<_> = distrs[1..].iter().map(|distr| Box::new(distr.clone()) ).collect();
        let mut base = distrs[0].clone();
        base.mix = Some(mix);
        base
    }

}

impl Into<serde_json::Value> for MultiNormal {

    fn into(self) -> serde_json::Value {
        let mu = crate::model::vector_to_value(&self.mu);
        let sigma = crate::model::matrix_to_value(&self.sigma);
        let mut child = Map::new();
        child.insert(String::from("mean"), mu);
        child.insert(String::from("cov"), sigma);
        let mut parent = Map::new();
        parent.insert(String::from("multinormal"), Value::Object(child));
        Value::Object(parent)
    }

}

#[test]
fn suff_log_prob() {
    let y = DMatrix::from_element(4, 4, 2.0);
    let suff = MultiNormal::sufficient_stat((&y).into());
    println!("{}", suff);
}

#[test]
fn regression() {
    let norm = Normal::new(100, None, None);
    let noise = [norm.sample(), norm.sample(), norm.sample()];
    let beta = DVector::from_column_slice(&[1., 1., 1.]);
    let x1 = DVector::from_fn(100, |i,_| beta[0] * i as f64 + noise[0][i]);
    let x2 = DVector::from_fn(100, |i,_| beta[1] * i as f64 + noise[1][i]);
    let x3 = DVector::from_fn(100, |i,_| beta[2] * i as f64 + noise[2][i]);
    let mut y = norm.sample().column(0).clone_owned();
    y += &x1;
    y += &x2;
    y += &x3;
    let x = DMatrix::from_columns(&[x1, x2, x3]);
    let x = utils::append_intercept(x.clone());

    let beta = MultiNormal::linear_mle(&x, &y);
    println!("Linear MLE = {}", beta.mean());

    // let yh = x * beta.mean();
}

/*mod estimation {

    use super::*;

    /// Returns the normalized data, and the mean and variance normalization factors.
    pub fn normalize(data : &DMatrix<f32>) -> (DMatrix<f32>, DVector<f32>, DVector<f32>) {
        let ncols = data.ncols();
        let mut means = DVector::<f32>::from_element(ncols, 0.0);
        let mut vars = DVector::<f32>::from_element(ncols, 0.0);
        let norm_cols : Vec<DVector<f32>> =
            data.column_iter().zip(means.iter_mut()).zip(vars.iter_mut())
                .map(|((col, m), v)| {
                let (mean, var) = mean_variance(&col.into());
                *m = mean;
                *v = var;
                (col.add_scalar(-mean)) / var
            }).collect();
        let norm_data = DMatrix::<f32>::from_columns(&norm_cols[..]);
        (norm_data, means, vars)
    }

    /*/// Weighted least squares estimation. Assumes elements are centered and
    /// correctly scaled.
    /// augment_elems : Augment the responses matrix using those factors.
    /// Response matrix will be assumed to assume value of zero over those elements.
    /// (useful for certain Bayesian estimation procedures).
    /// weights : Vector with the diagonal of the weights matrix.
    /// Returns: Parameter matrix, residual matrix.
    /// Perhaps pass augmentationt to a separate routine that takes a regularization
    /// factor as parameter (functionality of WLS does not depend on whether elements
    /// are augment or not, as long as they are normalized).
    pub fn wls(
        predictors : &DMatrix<f32>,
        responses : &DVector<f32>,
        augment_elems : Option<&DMatrix<f32>>,
        weights : Option<&DVector<f32>>,
    ) -> Result<(DVector<f32>, DVector<f32>),&'static str> {
        let mut predictors = predictors.clone();
        let mut responses = responses.clone();
        if let Some(mut aug) = augment_elems {
            let mut aug = aug.clone();
            let mut predictors = predictors.clone().insert_rows(predictors.nrows(), aug.nrows(), 0.0);
            let responses = responses.clone().insert_rows(responses.clone().nrows(), aug.nrows(), 0.0);
            predictors.row_iter_mut()
                .skip(responses.nrows())
                .zip(aug.row_iter_mut())
                .map(|(mut r1, r2)| r1 = r2.into());
        }

        let cross_prods = match weights {
            Some(w) => {
                predictors.clone().transpose() *
                    DMatrix::<f32>::from_diagonal(&w) *
                    predictors.clone()
            },
            None => {
                predictors.clone().transpose() * predictors.clone()
            }
        };

        let cross_qr = cross_prods.qr();
        let dep_prods = predictors.clone() * responses.clone();
        if let Some(params) = cross_qr.solve(&dep_prods) {
            let residuals = (predictors * params.clone()) - responses;
            Ok((params, residuals))
        } else {
            Err("Error solving QR decomposition")
        }
    }*/
}*/

/*#[test]
pub fn gls() {
    let y : DVector<f64> = DVector::from_vec(vec![1.0, 2.2, 3.1, 4.9, 4.43]);
    let x : DMatrix<f64> = DMatrix::from_vec(5,2,
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 4.2, 1.3, 2.7, 0.4, 3.06]
    );
    let yc = y.add_scalar(-y.mean());
    let dist = yc.clone() * yc.clone().transpose();
    //println!("{}", dist);
    //let cov = dist.clone().transpose() * dist;
    let mut cov = DMatrix::zeros(y.nrows(), y.nrows());
    cov.set_diagonal(&DVector::from_element(y.nrows(), 1.));
    cov = cov.scale(5.);
    let ols_est = OLS::estimate(y.clone(), Some(x.clone()));
    let est = GLS::estimate(yc, cov, Some(x));
    //assert!(ols_est == ident_est)
    println!("{:?}", est)
}

#[test]
fn wls() {
    let y : DVector<f64> = DVector::from_vec(vec![1.0, 2.2, 3.1, 4.9, 4.43]);
    let x : DMatrix<f64> = DMatrix::from_vec(5,2,
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 4.2, 1.3, 2.7, 0.4, 3.06]
    );
    let ols = OLS::estimate(y.clone(), Some(x.clone()));
    let var = DVector::from_element(y.nrows(), 1.);
    let wls = WLS::estimate(y.clone(), var.clone(), x.clone());
    //println!("{}", var.)
}*/

/*/// This is valid only for independent normals. But since .condition(.)
/// takes ownership of a value, there is no way a user will build
/// dependent normal distributions.
impl Add<MultiNormal, Output=MultiNormal> for MultiNormal {

}

impl Mul<DMatrix<f64>, Output=Normal> for MultiNormal {

}

impl Mul<f64, Output=MultiNormal> for MultiNormal {

}

impl Add<DVector<f64>, Output=MultiNormal for MultiNormal {

}

// Add and subtract covariances.
impl Add<MultiNormal> for MultiNormal {

}

impl Sub<MultiNormal> for MultiNormal {

}*/

/*pub fn flatten_matrix(m : DMatrix<f64>) -> Vec<Vec<f64>> {
    let mut rows = Vec::new();
    for _ in 0..m.nrows() {
        let v : Vec<f64> = m.remove_row(0).data.into();
        rows.push(v);
    }
    rows
}*/

/*/// Distribution resulting from either a marginalization or conditioning operation on
/// a multivariate normal. It implements the descriptive interface of distribution,
/// but it never yields any factors. Returned by mn.marginal(.) (in which case cond will
/// be None) or mn.conditional(a) (in which case cond will be a).
pub struct NormalSlice<'a> {
    mn : &'a MultiNormal
    offset : usize,
    dim : usize,
    cond : Option<DVector<f64>>
}

impl<'a> NormalSlice<'a> {

    /// Creates a new full multivariate from this slice
    pub fn clone_multivariate(&self) -> MultiNormal {
        unimplemented!()
    }

    /// Creates a new univariate from this slice, if it has dimension one.
    pub fn clone_univariate(&self) -> Option<Normal> {
        unimplemented!()
    }

}*/

/*impl Distribution for NormalSlice
    where Self : Debug + Display
{

    /// Returns the expected value of the distribution, which is a function of
    /// the current parameter vector.
    fn mean(&self) -> &DVector<f64> {
        unimplemented!()
    }

    // TODO transform API to:
    // "natural" is always the linear (or scaled in the normal case) parameter;
    // "canonical" is always the non-linear (or unscaled in the normal case) parameter.

    // view_natural() / set_natural()
    // view_canonical() / set_canonical()

    /// Acquires reference to internal parameter vector; either in
    /// natural form (eta) or canonical form (theta).
    fn view_parameter(&self, natural : bool) -> &DVector<f64> {
        unimplemented!()
    }

    fn set_natural<'a>(&'a mut self, eta : &'a mut dyn Iterator<Item=&'a f64>) {
        unimplemented!()
    }

    /// Set internal parameter vector at the informed value; either passing
    /// the natural form (eta) or the canonical form (theta).
    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, natural : bool) {
        unimplemented!()
    }

    /// Returns the global maximum of the log-likelihood, which is a function
    /// of the current parameter vector. Note that
    /// for bounded distributions (Gamma, Beta, Dirichlet), this value might just the the
    /// the parameter domain inferior or superior limit.
    fn mode(&self) -> DVector<f64> {
        unimplemented!()
    }

    /// Returns the dispersion of the distribution. This is a simple function of the
    /// parameter vector for discrete distributions; but is a constant or conditioning
    /// factor for continuous distributions. Returns the diagonal of the covariance
    /// matrix for continuous distributions.
    fn var(&self) -> DVector<f64> {
        unimplemented!()
    }

    /// Returns None for univariate distributions; Returns the positive-definite
    /// covariance matrix (inverse of precision matrix) for multivariate distributions
    /// with independent scale parameters.
    fn cov(&self) -> Option<DMatrix<f64>> {
        unimplemented!()
    }

    /// Returns the inverse-covariance (precision) for multivariate
    /// distributinos; Returns None for univariate distributions.
    fn cov_inv(&self) -> Option<DMatrix<f64>> {
        unimplemented!()
    }

    fn corr(&self) -> Option<DMatrix<f64>> {
        // D^{-1/2} self.cov { D^{-1/2} }
        unimplemented!()
    }

    // Returns a histogram if this distribution is univariate
    fn histogram(&self, bins : usize) -> Option<Histogram> {
        None
    }

    // Returns a kernel density estimate if this distribution is univariate
    fn smooth(&self, kernel : Kernel) -> Option<Density> {
        None
    }

    // Returns a symmetric interval around the mean if this distribution is univariate and symmetric
    fn interval(&self) -> Option<Interval> {
        None
    }

    // Returns a highest-density interval if this distribution is univariate
    fn hdi(&self) -> Option<HDI> {
        None
    }

    // fn observations(&self) -> Option<&DMatrix<f64>> {
    //    unimplemented!()
    // }

    // Effectively updates the name of this distribution.
    // fn rename(&mut self);

    /// Evaluates the log-probability of the sample y with respect to the current
    /// parameter state, and optionally by transforming the random variable by the matrix x.
    /// This method just dispatches to a sufficient statistic log-probability
    /// or to a conditional log-probability evaluation depending on the conditioning factors
    /// of the implementor. The samples at matrix y are assumed to be independent over rows (or at least
    /// conditionally-independent given the current factor graph). Univariate factors require that
    /// y has a single column; Multivariate factors require y has a number of columns equal to the
    /// distribution parameter vector.
    fn log_prob(&self /*, y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>*/ ) -> Option<f64> {
        unimplemented!()
    }

    fn sample_into(&self, dst : DMatrixSliceMut<'_,f64>) {
        unimplemented!()
    }

    fn dyn_factors(&self) -> (Option<&dyn Distribution>, Option<&dyn Distribution>) {
        (None, None)
    }

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Distribution>, Option<&mut dyn Distribution>) {
        (None, None)
    }

}*/

// TODO implement TryInto<Normal> if multinormal has a single dimension (useful for EM algorithm)
