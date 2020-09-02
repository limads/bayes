use nalgebra::*;
use super::*;
use serde::{Serialize, Deserialize};
use std::f64::consts::PI;
use std::fmt::{self, Display};
use std::default::Default;
use std::ops::{SubAssign, MulAssign};
use std::convert::TryInto;
use crate::sim::RandomWalk;

#[derive(Debug, Clone, Serialize, Deserialize)]
enum CovFunction {
    None,
    Log,
    Logit
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinearOp {

    pub scale : DMatrix<f64>,

    pub scale_t : DMatrix<f64>,

    pub shift : DVector<f64>,

    pub lin_mu : DVector<f64>,

    pub lin_scaled_mu : DVector<f64>,

    pub lin_sigma_inv : DMatrix<f64>,

    pub lin_log_part : DVector<f64>,

    /*pub transf_sigma_inv : Option<DMatrix<f64>>,

    /// For situations when the covariance is a known
    /// function of the mean parameter vector, usually
    /// because we are making inferences conditional on an
    /// ML estimate of the scale parameter.
    pub cov_func : CovFunction*/
}

impl Default for LinearOp {

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

}

impl LinearOp {

    pub fn new_from_shift(shift : DVector<f64>) -> Self {
        let mut op : LinearOp = Default::default();
        let mut scale = DMatrix::zeros(shift.nrows(), shift.nrows());
        scale.set_diagonal(&DVector::from_element(shift.nrows(), 1.));
        op.shift = shift;
        op.scale_t = scale.clone().transpose();
        op.scale = scale;
        op
    }

    pub fn new_from_scale(scale : DMatrix<f64>) -> Self {
        let mut op : LinearOp = Default::default();
        op.shift = DVector::from_element(scale.nrows(), 0.);
        op.scale_t = scale.clone().transpose();
        op.scale = scale;
        op
    }

    pub fn update(&mut self, mu : &DVector<f64>, sigma_inv : &DVector<f64>) {
        self.lin_mu = self.scale.clone() * mu;
        self.lin_sigma_inv = self.scale.clone() * sigma_inv * &self.scale_t;
        self.lin_scaled_mu = self.lin_sigma_inv.clone() * &self.lin_mu
    }

    pub fn output(&self) -> (&DVector<f64>, &DMatrix<f64>) {
        (&self.lin_mu, &self.lin_sigma_inv)
    }

}

/// Multivariate normal parametrized by μ (px1) and Σ (pxp).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiNormal {
    mu : DVector<f64>,

    /// (sigma_inv * mu)^T
    scaled_mu : DVector<f64>,

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

    scale_factor : Option<Wishart>,

    /// This is a single-element vector. Unlike the univariate distributions (Poisson, Bernoulli)
    /// that hold conditional expectations and thus have a log-partition with the same size as eta,
    /// the multinormal holds a single parameter value, with its corresponding scalar log-partition.
    log_part : DVector<f64>,

    rw : Option<RandomWalk>,

    // Vector of scale parameters; against which the Wishart factor
    // can be updated. Will be none if there is no scale factor.
    // suff_scale : Option<DVector<f64>>
}

impl MultiNormal {

    pub fn invert_scale(s : &DMatrix<f64>) -> DMatrix<f64> {
        let s_qr = QR::<f64, Dynamic, Dynamic>::new(s.clone());
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

    pub fn new(mu : DVector<f64>, sigma : DMatrix<f64>) -> Self {
        // let suff_scale = None;
        let log_part = DVector::from_element(1, 0.0);
        let sigma_inv = Self::invert_scale(&sigma);
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
            mu : mu.clone(),
            sigma_inv,
            loc_factor: None,
            scale_factor : None,
            op : None,
            log_part,
            rw : None,
            scaled_mu : mu.clone()
        };
        norm.set_parameter(mu.rows(0, mu.nrows()), true);
        norm.set_cov(sigma.clone());
        norm
    }

    pub fn scale(&mut self, scale : DMatrix<f64>) {
        match self.op {
            Some(ref mut op) => {
                op.scale = scale;
            },
            None => {
                self.op = Some(LinearOp::new_from_scale(scale));
            }
        }
    }

    pub fn shift(&mut self, shift : DVector<f64>) {
        match self.op {
            Some(ref mut op) => {
                op.shift = shift;
            },
            None => {
                self.op = Some(LinearOp::new_from_shift(shift));
            }
        }
    }

    fn multinormal_log_part(mu : DVectorSlice<'_, f64>, scaled_mu : DVectorSlice<'_, f64>, cov : &DMatrix<f64>, prec : &DMatrix<f64>) -> f64 {
        let sigma_lu = LU::new(cov.clone());
        let sigma_det = sigma_lu.determinant();

        let prec_lu = LU::new(prec.clone());
        let prec_det = prec_lu.determinant();
        //let p_mu_cov = -0.25 * scaled_mu.clone().transpose() * cov * &scaled_mu;

        //println!("sigma det ln: {}", sigma_det.ln());
        -0.5 * (mu.clone().transpose() * scaled_mu)[0] - 0.5 * sigma_det.ln()
        // p_mu_cov[0] - 0.5*sigma_det.ln()
    }

    /// Returns the reduced multivariate normal [ix, ix+n) by marginalizing over
    /// the remaining indices [0, ix) and [ix+n,p). This requires a simple copy
    /// of the corresponding entries. The scale and shift matrices are also
    /// sliced at the respective entries.
    pub fn marginal(&self, ix : usize, n : usize) -> MultiNormal {
        let size = n - ix;
        let mut marg = self.clone();
        // Modify mu
        // Modify sigma
        // Modify scale factors
        marg
    }

    /// Returns the reduced multivariate normal [0, n) by conditioning
    /// over entries [n,p) at the informed value. This operation condition
    /// on a user-supplied constant and has nothing to do with the
    /// Conditional<Distribution> trait, which is used for conditioning over
    /// other random variables.
    pub fn conditional(&self, value : DVector<f64>) -> MultiNormal {
        let mut cond = self.marginal(0, value.nrows());
        cond.cond_update_from(&self, value);
        cond
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

    /// Changes self by assuming joint normality with another
    /// independent distribution (extends self to have a block-diagonal
    /// covariance composed of the covariance of self (top-left block)
    /// with the covariance of other (bottom-right block).
    pub fn joint(&mut self, other : MultiNormal) {
        unimplemented!()
    }

    pub fn linear_mle(x : &DMatrix<f64>, y : &DVector<f64>) -> Self {
        let n = x.nrows() as f64;
        let ols = ls::OLS::estimate(&y, &x);
        let mu = ols.beta;
        let cov = (x.clone().transpose() * x).scale(ols.err.unwrap().sum() / n);
        Self::new(mu, cov)
    }

    pub fn set_cov(&mut self, cov : DMatrix<f64>) {
        let prec = Self::invert_scale(&cov);
        self.scaled_mu = prec.clone() * &self.mu;
        self.sigma_inv = prec;
        let mu = self.mu.clone();
        self.update_log_partition(mu.rows(0, mu.nrows()));
    }

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

}

impl ExponentialFamily<Dynamic> for MultiNormal {

    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        DVector::from_element(1, (2. * PI).powf( - (y.ncols() as f64) / 2. ) )
    }

    /// Returs the matrix [sum(yi) | sum(yi yi^T)]
    /// TODO if there is a constant scale factor, the sufficient statistic
    /// is the sample row-sum. If there is a random scale factor (wishart) the sufficient
    /// statistic is the matrix [rowsum(x)^T sum(x x^T)]
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
        let scaled_mu = match self.op {
            Some(ref op) => &(op.lin_scaled_mu),
            None => &self.scaled_mu
        };
        let sigma_inv = match self.op {
            Some(ref op) => &(op.lin_sigma_inv),
            None => &self.sigma_inv
        };
        let mut lp = 0.0;
        lp += scaled_mu.dot(&t.column(0));
        let t_cov = t.columns(1, t.ncols() - 1);
        for (s_inv_row, tc_row) in sigma_inv.row_iter().zip(t_cov.row_iter()) {
            lp += (-0.5) * s_inv_row.dot(&tc_row);
        }
        let log_part = match self.op {
            Some(ref op) => &op.lin_log_part,
            None => &self.log_partition()
        };
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
    fn update_log_partition<'a>(&'a mut self, eta : DVectorSlice<'_, f64>) {
        // TODO update eta parameter here.
        let cov = Self::invert_scale(&self.sigma_inv);
        let log_part = Self::multinormal_log_part(
            eta,
            self.scaled_mu.rows(0, self.scaled_mu.nrows()),
            &cov,
            &self.sigma_inv
        );
        self.log_part = DVector::from_element(1, log_part);
        if let Some(ref mut op) = self.op {
            // The new lin_mu should have already been set at self.set_parameter()
            let lin_mu = op.lin_mu.clone_owned();
            let lin_sigma_inv = op.lin_sigma_inv.clone_owned();
            let lin_cov = Self::invert_scale(&lin_sigma_inv);

            // pass scaled_lin_mu here instead
            let lin_log_part = Self::multinormal_log_part(
                lin_mu.rows(0, lin_mu.nrows()),
                lin_mu.rows(0, lin_mu.nrows()),
                &lin_cov,
                &lin_sigma_inv
            );
            op.lin_log_part = DVector::from_element(1, lin_log_part);
        }
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

    fn view_parameter(&self, _natural : bool) -> &DVector<f64> {
        // see mu; vs. see sigma_inv*mu
        unimplemented!()
    }

    /// Set parameter should verify if there is a scale factor. If there is,
    /// also sets the eta of those factors and write the new implied covariance
    /// into self. Only after the new covariance is set to self, call self.update_log_partition().
    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, _natural : bool) {
        self.mu.copy_from(&p.column(0));
        if let Some(ref mut op) = self.op {
            op.lin_mu = &op.scale * p.clone_owned();
            /*match op.cov_func {
                CovFunction::Log => {
                    // op.transf_sigma_inv = Some(self.sigma_inv[(0, 0)] * op.scale.clone() * op.scale.clone().transpose());
                },
                CovFunction::Logit => {

                }
            }
            op.lin_mu = op.scale.clone() * p.column(0);
            op.lin_sigma_inv = op.scale.clone() * self.sigma_inv * op.scale.clone().transpose();*/
        }
        self.scaled_mu = self.sigma_inv.clone() * &self.mu;
        self.update_log_partition(p);
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.mu
    }

    fn mode(&self) -> DVector<f64> {
        self.mu.clone()
    }

    fn var(&self) -> DVector<f64> {
        self.cov().unwrap().diagonal()
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        Some(Self::invert_scale(&self.sigma_inv))
    }

    fn cov_inv(&self) -> Option<DMatrix<f64>> {
        Some(self.sigma_inv.clone())
    }

    fn log_prob(&self, y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>) -> f64 {
        let t = Self::sufficient_stat(y);
        let lp = self.suf_log_prob(t.slice((0, 0), (t.nrows(), t.ncols())));
        let loc_lp = match &self.loc_factor {
            Some(loc) => {
                let mu_row : DMatrix<f64> = DMatrix::from_row_slice(self.mu.nrows(), 1, self.mu.data.as_slice());
                loc.log_prob(mu_row.slice((0, 0), (0, self.mu.nrows())), None)
            },
            None => 0.0
        };
        let scale_lp = match &self.scale_factor {
            Some(scale) => {
                let sinv_diag : DVector<f64> = self.sigma_inv.diagonal().clone_owned();
                scale.log_prob(sinv_diag.slice((0, 0), (sinv_diag.nrows(), 1)), None)
            },
            None => 0.0
        };
        lp + loc_lp + scale_lp
    }

    fn sample_into(&self, _dst : DMatrixSliceMut<'_, f64>) {
        unimplemented!()
    }

}

impl Posterior for MultiNormal {

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Posterior>, Option<&mut dyn Posterior>) {
        let loc = self.loc_factor.as_mut().map(|lf| lf.as_mut() as &mut dyn Posterior);
        let scale = self.scale_factor.as_mut().map(|sf| sf as &mut dyn Posterior);
        (loc, scale)
    }

    fn approximation_mut(&mut self) -> Option<&mut MultiNormal> {
        Some(self)
    }

    fn approximation(&self) -> Option<&MultiNormal> {
        Some(self)
    }

    fn trajectory(&self) -> Option<&RandomWalk> {
        self.rw.as_ref()
    }

    fn trajectory_mut(&mut self) -> Option<&mut RandomWalk> {
        self.rw.as_mut()
    }

}

impl Likelihood<Dynamic> for MultiNormal {

    /// Returns the distribution with the parameters set to its
    /// gaussian approximation (mean and standard error).
    fn mle(y : DMatrixSlice<'_, f64>) -> Self {
        let n = y.nrows() as f64;
        let suf = Self::sufficient_stat(y);
        let mut mu : DVector<f64> = suf.column(0).clone_owned();
        mu.unscale_mut(n);
        let mut sigma = suf.remove_column(0);
        sigma.unscale_mut(n);
        Self::new(mu, sigma)
    }

    fn visit_factors<F>(&mut self, f : F) where F : Fn(&mut dyn Posterior) {
        if let Some(ref mut loc) = self.loc_factor {
            f(loc.as_mut());
            loc.visit_post_factors(&f as &dyn Fn(&mut dyn Posterior));
        }
        if let Some(ref mut scale) = self.scale_factor {
            f(scale);
            scale.visit_post_factors(&f as &dyn Fn(&mut dyn Posterior));
        }
    }
}

impl Estimator<MultiNormal> for MultiNormal {

    fn fit<'a>(&'a mut self, y : DMatrix<f64>, x : Option<DMatrix<f64>>) -> Result<&'a MultiNormal, &'static str> {
        let n = y.nrows() as f64;
        match (&mut self.loc_factor, &mut self.scale_factor) {
            (Some(ref mut norm_prior), Some(ref mut gamma_prior)) => {
                unimplemented!()
            },
            (Some(ref mut norm_prior), None) => {
                // Calculate sample centroid
                let suf = Self::sufficient_stat((&y).into());
                let mut mu_mle : DVector<f64> = suf.column(0).clone_owned();
                mu_mle.unscale_mut(n);

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
    }

}

impl Display for MultiNormal {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MNorm({})", self.mu.nrows())
    }

}

mod utils {

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

/// Least squares algorithms
mod ls {

    use super::*;

    /// Ordinary least square estimation.
    #[derive(Debug)]
    pub struct OLS {
        pub beta : DVector<f64>,
        pub err : Option<DVector<f64>>
    }

    impl OLS {

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
    /// covariance (diagonal precision)
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
    /// non-diagonal covariances (estimation under any error covariance).
    /// This covariance has an n x n structure and represents the possible
    /// correlations of all observations of dimensionality 1 with themselves. The solution is:
    /// Multiply X and y by the Cholesky factor of this matrix; Then solve OLS for the standardized
    /// quantities. The same equation for WLS can be applied here.
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

