use rand_distr;
use std::borrow::Borrow;
use rand_distr::{StandardNormal};
use crate::prob::*;
use std::iter::IntoIterator;
use std::default::Default;
pub use rand_distr::Distribution;
use rand::Rng;
use nalgebra::*;
use approx::*;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct Normal {

    loc : f64,

    scale : f64,

}

impl Normal {

    pub fn new(loc : f64, scale : f64) -> Self {
        Self { loc, scale }
    }

}

impl Default for Normal {

    fn default() -> Self {
        STANDARD_NORMAL.clone()
    }

}

/*impl Normal {

    fn sample<R>(&self, rng: &mut R) -> f64
    where
        R: rand::Rng + ?Sized,
        // StandardNormal : rand_distr::Distribution<f64>
        // StandardNormal : rand::distributions::Distribution<f64>
    {
        use rand::prelude::*;
        let z : f64 = rng.sample(rand_distr::StandardNormal);
        self.scale.sqrt() * (z + self.loc)
    }

}*/

// In a location-scale family, p(x|loc, scale) = (1/scale)*p_s((x - loc)/scale),
// where p_s is the distribution standard (loc=0 and scale=1).

// cargo test -- normal --nocapture
#[test]
fn normal() {
    let n = Normal::new(1.0, 10.0);
    abs_diff_eq!(n.log_probability(3.), -2.270231079701696 );
}

impl Univariate for Normal { }

// 1 / (2*pi) (normal base measure)
const INV_2_PI : f64 = 0.1591549430918953357688837633725143620344596457404564487476673440;

// (2*pi).ln()
const LN_2_PI : f64 = 1.8378770664093454835606594728112352797227949472755668256343030809;

impl Exponential for Normal {

    fn log_probability(&self, y : f64) -> f64 {
        normal_log_prob(y, self.loc, self.scale.sqrt())
    }

    fn base_measure(&self) -> f64 {
        INV_2_PI
    }

    fn location(&self) -> f64 {
        self.loc
    }

    fn link(avg : f64) -> f64 {
        avg
    }

    fn link_inverse(avg : f64) -> f64 {
        avg
    }

    fn scale(&self) -> Option<f64> {
        Some(self.scale)
    }

    // fn partition(&self) -> f64 {
    //    unimplemented!()
    // }

    fn log_partition(&self) -> f64 {
        self.loc.powf(2.) / (2.*self.scale) + self.scale.sqrt().ln()
    }

}

impl rand_distr::Distribution<f64> for Normal {

    fn sample<R>(&self, rng: &mut R) -> f64
    where
        R: rand::Rng + ?Sized
    {
        // If S is an integer seed and a, b c are constants,
        // r = (a S + b) % c is a new random variate (divide a S + b by c).
        // If r1, r2 are uniformly-distributed variates, sqrt(-2 log r1) cos(2 \pi r2) is
        // a standard normal variate (Smith, 1997).
        use rand::prelude::*;
        let z : f64 = rng.sample(rand_distr::StandardNormal);
        z * self.scale.sqrt() + self.loc
    }

}

/*impl Prior for Normal {

    fn prior(loc : f64, scale : Option<f64>) -> Self {
        Self { loc, scale : scale.unwrap_or(1.0), n : 1, factor : None }
    }

}*/

/*impl Posterior for Normal {

    fn size(&self) -> Option<usize> {
        Some(self.n)
    }

}*/

/*impl Likelihood for Normal {

    fn likelihood(sample : &[f64]) -> Joint<Self> {
        // let (loc, scale, n) = crate::calc::running::single_pass_sum_sum_sq(sample.iter());
        // Normal { loc, scale, n }
        Joint::<Normal>::from_slice(sample)
    }

}*/

/*/// Condition<MultiNormal> is implemented for [Normal] but not for Normal,
/// which gives the user some type-safety when separating regression AND mixture
/// from conjugate models. A regression model is Condition<MultiNormal> for [Normal],
/// and a mixture model is Condition<Categorical> for [Normal]. For mixture models,
/// the normals are set at their MLE, and we perform inference over the categorical
/// variable.
impl<'a> Condition<MultiNormal> for [Normal] {

    fn condition(&mut self, f : MultiNormal) -> &mut Self {
        unimplemented!()
    }

}*/

pub const STANDARD_NORMAL : Normal = Normal {
    loc : 0.0,
    scale : 1.0
};

/*impl rand::distributions::Distribution<f64> for Normal {

    // Can accept ThreadRng::default()
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand::prelude::*;
        let z : f64 = rng.sample(self.sampler);
        self.scale.sqrt() * (z + self.loc)
    }

}*/

// based on stats::dnorm.ipp
fn normal_log_prob(x : f64, mu : f64, stddev : f64) -> f64 {
    std_normal_log_prob((x - mu) / stddev, stddev)
}

// based on stats::dnorm.ipp
fn std_normal_log_prob(z : f64, stddev : f64) -> f64 {
    -0.5 * LN_2_PI - stddev.ln() - z.powf(2.0) / 2.0
}

// Based on the statslib impl
pub(crate) fn multinormal_log_prob(x : &DVector<f64>, mean : &DVector<f64>, cov : &DMatrix<f64>) -> f64 {

    // This term can be computed at compile time if VectorN is used. Or we might
    // keep results into a static array of f64 and just index it with x.nrows().
    let partition = -0.5 * x.nrows() as f64 * LN_2_PI;

    let xc = x.clone() - mean;

    let cov_chol = Cholesky::new(cov.clone()).unwrap();

    // x^T S^-1 x
    let mahalanobis = xc.transpose().dot(&cov_chol.solve(&xc));

    partition - 0.5 * (cov_chol.determinant().ln() + mahalanobis)
}

impl Joint<Normal> {

    pub fn mean(&self) -> &DVector<f64> {
        &self.loc
    }

    pub fn probability(&self, x : &DVector<f64>) -> f64 {
        self.log_probability(x).exp()
    }

    pub fn log_probability(&self, x : &DVector<f64>) -> f64 {
        multinormal_log_prob(x, self.mean(), self.scale.as_ref().unwrap())
    }

}

// When generating joint gaussian samples, add a small multiple of the identity \epsilon I to
// the covariance before inverting it, for numerical reasons.

/*

 pub fn univariate_mle(data : &[f64]) -> (f64, f64) {
        let (sum, sum_sq) = data.iter()
            .fold((0.0, 0.0), |accum, d| {
                (accum.0 + d, accum.1 + d.powf(2.))
            });
        let n = data.len() as f64;
        let mean = sum / (n as f64);
        (mean, sum_sq / (n as f64) - mean.powf(2.))
    }

impl<'a> Estimator<'a, &'a Normal> for Normal {

    type Algorithm = ();

    type Error = &'static str;

    //fn predict<'a>(&'a self, cond : Option<&'a Sample/*<'a>*/>) -> Box<dyn Sample/*<'a>*/> {
    //    unimplemented!()
    //}

    /*fn take_posterior(self) -> Option<Normal> {
        unimplemented!()
    }

    fn view_posterior<'a>(&'a self) -> Option<&'a Normal> {
        unimplemented!()
    }*/

    fn fit(&'a mut self, algorithm : Option<Self::Algorithm>) -> Result<&'a Normal, &'static str> {
        // self.observe_sample(sample);
        let prec1 = 1. / self.var()[0];

        match (&mut self.loc_factor, &mut self.scale_factor) {
            (NormalFactor::Conjugate(ref mut norm), Some(ref mut gamma)) => {
                unimplemented!()
            },
            (NormalFactor::Conjugate(ref mut norm), None) => {
                let y = self.obs.clone().unwrap();
                assert!(norm.mean().len() == 1, "Length of mean vector should be one");
                let n = y.nrows() as f64;
                let ys = y.column(0).sum();
                let mu0 = norm.mean()[0];
                let prec0 = 1. / norm.var()[0];

                // B = sigma_a^2 / (sigma_a^2 + sigma_b^2) is called the shrinkage factor,
                // since the posterior average is a weighted sum: B*\mu + (1. - B)*y
                let var_out = 1. / (prec0 + n*prec1);
                let mu_out =  var_out*(mu0*prec0 + ys*prec1);
                norm.set_parameter((&DVector::from_element(1, mu_out)).into(), true);
                norm.set_var(var_out);
                Ok(&(*norm))
            },
            _ => Err("Distribution does not have a conjugate location factor")
        }
    }

}

fn normal_sample(m : f64, sd : f64) -> f64 {
    use rand::prelude::*;
    let z : f64 = rand::thread_rng().sample(rand_distr::StandardNormal);
    sd * (z + m)
}

fn joint_log_prob(&self /*, y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>*/ ) -> Option<f64> {
        /*let eta = match self.current() {
            Some(eta) => eta,
            None => self.mu.rows(0, self.mu.nrows())
        };*/
        let loc_lp = match self.loc_factor {
            NormalFactor::Conjugate(ref loc) => loc.suf_log_prob(self.mu.slice((0, 0), (self.mu.nrows(), 1))),
            NormalFactor::Fixed(ref mn) => mn.suf_log_prob(self.mu.slice((0, 0), (self.mu.nrows(), 1))),
            NormalFactor::Empty => 0.
        };
        let scale_lp = match self.scale_factor {
            Some(ref scale) => scale.suf_log_prob(self.prec_suff.slice((0,0), (2, 1))),
            None => 0.
        };
        let y = self.obs.as_ref()?;
        let t = Self::sufficient_stat(y.slice((0, 0), (y.nrows(), 1)));
        let lp = self.cond_log_prob()?.sum() + loc_lp + scale_lp;
        Some(lp)
    }

    fn sample_into(&self, mut dst : DMatrixSliceMut<'_, f64>) {
        // assert!(dst.ncols() == 1);
        // for (i, m) in self.mu.iter().enumerate() {
        //    dst[(i,0)] = self.rng.normal(*m, var.sqrt() );
        // }

        let opt_mu = sample_natural_factor_boxed(self.view_fixed_values(), &self.loc_factor, self.n);
        let mu = opt_mu.as_ref().unwrap_or(&self.mu);
        let var = self.var()[0];
        let sd = var.sqrt();
        for (i, m) in mu.iter().enumerate() {
            dst[(i, 0)] = normal_sample(*m, sd);
        }
    }

impl ExponentialFamily<U1> for Normal
    where
        Self : Distribution
{

    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        DVector::from_element(y.nrows(),  1. / (2.*PI).sqrt())
    }

    fn grad(&self, y : DMatrixSlice<'_, f64>, x : Option<DMatrix<f64>>) -> DVector<f64> {
        let mut s = (y.column(0) - self.mean()).unscale(self.var()[0].sqrt() ).sum();
        // let mut s = (y.column(0) - self.mean()).unscale(self.var()[0] ).sum();
        DVector::from_element(1, s)
    }

    // TODO if there is a constant scale factor, the sufficient statistic
    // is the sample sum. If there is a random scale factor (gamma) the sufficient
    // statistic is the 2-dimensional vector [sum(x) sum(x^2)]
    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        assert!(y.ncols() == 1);
        let mut suf = DMatrix::zeros(2, 1);
        for y in y.column(0).iter() {
            suf[(0,0)] += y;
            suf[(1,0)] += y.powf(2.0);
        }
        suf
    }

    // suf_log_prob assumes dimensionality of mu vector is 1
    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {
        // assert!(self.log_partition().nrows() == 1);
        // assert!(self.mu.nrows() == 1);
        // assert!(self.scaled_mu.nrows() == 1);
        // assert!(t.ncols() == 1 && t.nrows() == 2);
        self.scaled_mu[0] * t[(0,0)] + self.minus_half_prec*t[(1,0)] - self.log_part[0]
    }

    /// ignore mu, using the scaled mu already set at the update.
    /// eta = [ prec*mu; (-1/2)*prec ]
    /// The log_prob as a function of scaled_mu (natural parameter over real line) is the last expression;
    /// The log_prob as a function of [scaled_mu, ln(scale)] (natural parameter over real plane) is the
    /// second expression. The normal with a known variance is a function of a scalar; the normal with unknown
    /// variance is a function of a plane, and we must disambiguate between those two cases. The first case
    /// defines a parabola over the scaled_mu line; the second case defines a convex surface over the real plane.
    /// For the second case, the log-likelihood parabola maximum of scaled_mu as a function of scale
    /// will only match the sample mean when the scale is fixed at the sample MLE conditional on the current mu.
    /// Re-fixing the scale at each iteration, however, will not lead to a convex likelihood but an exp-looking
    /// function. The correct scale maximum likelihood must be set at the beginning of the optimization (equal
    /// to chosing a slice at the mu-sigma plane that pass through the sigma MLE).
    fn update_log_partition<'a>(&'a mut self, /*_mu : DVectorSlice<'_, f64>*/ ) {

        // This is the expression for unknown variance
        self.log_part = self.mu.map(|m| m.powf(2.) / (2.*self.var()[0]) + self.var()[0].sqrt().ln() );

        // This is the expression for known variance
        // self.log_part = self.scaled_mu.map(|m| 0.5 * m.powf(2.0) )
        // self.log_part = self.scaled_mu.map(|m| 0.5 * m.powf(2.0) * self.prec_suff[1] )
    }

    fn log_partition<'a>(&'a self ) -> &'a DVector<f64> {
        &self.log_part
    }

    // eta = [sigma*mu, 1/sigma] vector
    fn link_inverse<S>(_eta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        // assert!(eta.nrows() == 2 && eta.ncols() == 1);
        // DVector::from_column_slice(&[-0.5 * eta[0] / eta[1], -0.5 / eta[1]])
        unimplemented!()
    }

    // theta = [mu, sigma] vector
    fn link<S>(_theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        //assert!(theta.nrows() == 2 && theta.ncols() == 1);
        //DVector::from_column_slice(&[theta[0] / theta[1], -0.5 / theta[1]])
        unimplemented!()
    }

    /*fn update_grad(&mut self, _eta : DVectorSlice<'_, f64>) {
        unimplemented!()
    }

    fn grad(&self) -> &DVector<f64> {
        unimplemented!()
    }*/

}

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
            mix : Vec::new(),
            cat_factor : None
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
pub fn approx_pd<N : Copy + Scalar + RealField + From<f32>>(m : DMatrix<N>) -> DMatrix<N> {
    let symm_m = build_symmetric(m);
    let (u, mut diag) = spectral_dec(symm_m);
    diag.iter_mut().for_each(|d| *d = if *d >= N::from(0.0) { *d } else { N::from(0.0) });
    let diag_m = DMatrix::from_diagonal(&diag);
    let u_t = u.transpose();
    u * diag_m * u_t
}
*/