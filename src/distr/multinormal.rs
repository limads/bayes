use nalgebra::*;
use super::*;
// use std::fmt::{self, Display};
use serde::{Serialize, Deserialize};
// use super::Gamma;
use std::f64::consts::PI;
// use crate::sim::*;
// use std::ops::MulAssign;
use std::fmt::{self, Display};
use std::default::Default;

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

    log_part : DVector<f64>,

    // Vector of scale parameters; against which the Wishart factor
    // can be updated. Will be none if there is no scale factor.
    // suff_scale : Option<DVector<f64>>
}

impl MultiNormal {

    fn _from_approximation(mut eta : DMatrix<f64>, lp_traj : DVector<f64>) -> Self {
        assert!(eta.ncols() == lp_traj.nrows());
        let mu_approx = eta.column(eta.ncols()).clone_owned();
        let eta_c = eta.clone();
        let eta_iter = eta.column_iter_mut().skip(1).zip(eta_c.column_iter()).enumerate();
        for (i, (mut eta_curr, eta_last)) in eta_iter {
            eta_curr -= eta_last;
            eta_curr.unscale_mut(lp_traj[i+1] - lp_traj[i]);
        }
        let eta = eta.remove_column(0);
        let eta_t = eta_c.remove_column(0).transpose();
        let prec_approx = eta * eta_t;
        let sigma_approx = Self::invert_scale(&prec_approx);
        MultiNormal::new(mu_approx, sigma_approx)
    }

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
            scaled_mu : mu.clone()
        };
        norm.update_log_partition(mu.rows(0, mu.nrows()));
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

    fn multinormal_log_part(mu : DVectorSlice<'_, f64>, cov : &DMatrix<f64>) -> f64 {
        let sigma_lu = LU::new(cov.clone());
        let sigma_det = sigma_lu.determinant();
        let p_mu_cov = -0.25 * mu.clone().transpose() * cov * &mu;
        p_mu_cov[0] - 0.5*sigma_det.ln()
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
    /// over entries [n,p).
    pub fn conditional(&self, n : usize) -> MultiNormal {
        unimplemented!()
    }

    /// Changes self by assuming joint normality with another
    /// independent distribution (extends self to have a block-diagonal
    /// covariance composed of the covariance of self (top-left block)
    /// with the covariance of other (bottom-right block).
    pub fn joint(&mut self, other : MultiNormal) {
        unimplemented!()
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
        let log_part = Self::multinormal_log_part(eta, &cov);
        self.log_part = DVector::from_element(1, log_part);
        if let Some(ref mut op) = self.op {
            // The new lin_mu should have already been set at self.set_parameter()
            let lin_mu = op.lin_mu.clone_owned();
            let lin_sigma_inv = op.lin_sigma_inv.clone_owned();
            let lin_cov = Self::invert_scale(&lin_sigma_inv);
            let lin_log_part = Self::multinormal_log_part(lin_mu.rows(0, lin_mu.nrows()), &lin_cov);
            op.lin_log_part = DVector::from_element(1, lin_log_part);
        }
        //let sigma_lu = LU::new(cov.clone());
        //let sigma_det = sigma_lu.determinant();
        //let p_eta_cov = -0.25 * eta.clone().transpose() * cov * &eta;
        //self.log_part = DVector::from_element(1, p_eta_cov[0] - 0.5*sigma_det.ln())
    }

    // eta = [sigma_inv*mu,-0.5*sigma_inv] -> [mu|sigma]
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

    fn log_prob(&self, y : DMatrixSlice<f64>) -> f64 {
        let t = Self::sufficient_stat(y);
        let lp = self.suf_log_prob(t.slice((0, 0), (t.nrows(), t.ncols())));
        let loc_lp = match &self.loc_factor {
            Some(loc) => {
                let mu_row : DMatrix<f64> = DMatrix::from_row_slice(self.mu.nrows(), 1, self.mu.data.as_slice());
                loc.log_prob(mu_row.slice((0, 0), (0, self.mu.nrows())))
            },
            None => 0.0
        };
        let scale_lp = match &self.scale_factor {
            Some(scale) => {
                let sinv_diag : DVector<f64> = self.sigma_inv.diagonal().clone_owned();
                scale.log_prob(sinv_diag.slice((0, 0), (sinv_diag.nrows(), 1)))
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

    fn set_approximation(&mut self, _m : MultiNormal) {
        unimplemented!()
    }

    fn approximation(&self) -> Option<&MultiNormal> {
        unimplemented!()
    }

}

impl Display for MultiNormal {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MNorm({})", self.mu.nrows())
    }

}

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

}


*/

