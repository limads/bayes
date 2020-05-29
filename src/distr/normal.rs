use crate::distr::*;
use crate::distr::gamma::Gamma;
use rand_distr;
use rand;
use std::f64::consts::PI;
// use std::ops::Add;
use crate::sim::*;
// use std::ops::AddAssign;

#[derive(Debug)]
pub struct Normal {

    // Location parameter, against which location factor is
    // evaluated.
    mu : DVector<f64>,

    // Variance parameter.
    // var : Variance,

    joint_ix : (usize, usize),

    loc_factor : Option<Box<Normal>>,

    scale_factor : Option<Gamma>,

    log_part : DVector<f64>,

    eta_traj : Option<EtaTrajectory>,

    /// Holds [log(1/sigma); 1/sigma], against which
    /// any scale factors (if present) are evaluated.
    prec_suff : DVector<f64>,

    // When updating the scale, also update this as
    // the second element to the sufficient statistic vector
    // and pass it to the gamma factor.
    // inv_scale_log
}

impl Normal {

    /// Distribution conditional on the given scale factor for the precision.
    /// All log-probabilities are evaluated conditional against the parameter
    /// for this factor; All sampling is conditional on a sampled value for this
    /// factor.
    pub fn new(n : usize, loc : Option<f64>, scale : Option<f64>) -> Self {
        let mu = DVector::from_element(n, loc.unwrap_or(0.0));
        let joint_ix = (0, mu.nrows());
        let loc_factor = None;
        let scale_factor = None;
        let scale = scale.unwrap_or(1.);
        let prec_suff = DVector::from_column_slice(&[1. / scale, (1. / scale).ln()]);
        let eta_traj = None;
        let log_part = mu.map(|e| e.powf(2.) / 2. );
        Self{ mu, joint_ix, loc_factor, eta_traj, prec_suff, scale_factor, log_part }
    }

    pub fn mle(y : DMatrixSlice<'_, f64>) -> (f64, f64) {
        assert!(y.ncols() == 1);
        let n = y.nrows() as f64;
        y.iter().fold((0.0, 0.0), |ys, y| (ys.0 + y / n, ys.1 + y.powf(2.) / n) )
    }

    // TODO move to ConditionalDistibution<Gama> implementation.
    /*/// Distribution conditioned on a constant scale factor that is not
    /// modelled probabilistically.
    pub fn new_scaled(loc : &[f64], scale : f64) -> Self {
        let mu = DVector::from_column_slice(loc);
        let factor = Factor::Empty;
        let joint_ix = (0, mu.nrows());
        let eta_traj = None;
        Self{ mu, var : Variance::Constant(scale), joint_ix, factor, eta_traj }
    }

    pub fn scale_factor<'a>(&'a self) -> Option<&'a Gamma> {
        match self.var {
            Variance::Constant(_) => None,
            Variance::Random(ref g) => Some(&g)
        }
    }*/
}

impl ExponentialFamily<U1> for Normal
    where
        Self : Distribution
{

    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        DVector::from_element(y.nrows(),  1. / (2.*PI).sqrt())
    }

    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        assert!(y.ncols() == 1);
        let mut suf = DMatrix::zeros(2, 1);
        for y in y.column(0).iter() {
            suf[(0,0)] += y;
            suf[(1,0)] += y.powf(2.0);
        }
        suf
    }

    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {
        assert!(self.log_partition().nrows() == 1);
        assert!(self.mu.nrows() == 1);
        assert!(t.ncols() == 1 && t.nrows() == 2);
        self.prec_suff[1] * self.mu[0] * t[(0,0)] +
            self.prec_suff[1] * (-0.5)*self.prec_suff[1]*t[(1,0)] - self.log_part[0]
    }

    fn update_log_partition<'a>(&'a mut self, eta : DVectorSlice<'_, f64>) {
        assert!(eta.nrows() == 2 && eta.ncols() == 1);
        self.log_part[0] = -eta[0].powf(2.) /  4.*eta[1] - 0.5*(-2.*eta[1]).ln()
    }

    fn log_partition<'a>(&'a self ) -> &'a DVector<f64> {
        &self.log_part
    }

    // eta = [sigma*mu, 1/sigma] vector
    fn link_inverse<S>(eta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
    where S : Storage<f64, Dynamic, U1>
    {
        assert!(eta.nrows() == 2 && eta.ncols() == 1);
        DVector::from_column_slice(&[-0.5 * eta[0] / eta[1], -0.5 / eta[1]])
    }

    // theta = [mu, sigma] vector
    fn link<S>(theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        assert!(theta.nrows() == 2 && theta.ncols() == 1);
        DVector::from_column_slice(&[theta[0] / theta[1], -0.5 / theta[1]])
    }

    fn update_grad(&mut self, _eta : DVectorSlice<'_, f64>) {
        unimplemented!()
    }

    fn grad(&self) -> &DVector<f64> {
        unimplemented!()
    }

}

impl Distribution for Normal
    where Self : Sized
{

    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, _natural : bool) {
        self.mu = p.clone_owned();
    }

    fn view_parameter(&self, _natural : bool) -> &DVector<f64> {
        // see mu; vs. see sigma_inv*mu
        unimplemented!()
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.mu
    }

    fn mode(&self) -> DVector<f64> {
        self.mu.clone()
    }

    fn var(&self) -> DVector<f64> {
        let v = 1. / self.prec_suff[1];
        DVector::from_element(1, v)
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        None
    }

    fn log_prob(&self, y : DMatrixSlice<f64>) -> f64 {
        /*let eta = match self.current() {
            Some(eta) => eta,
            None => self.mu.rows(0, self.mu.nrows())
        };*/
        let loc_lp = match self.loc_factor {
            Some(ref loc) => loc.log_prob(self.mu.slice((0, 0), (self.mu.nrows(), 1))),
            None => 0.
        };
        let scale_lp = match self.scale_factor {
            Some(ref scale) => scale.log_prob(self.prec_suff.slice((0,0), (2, 1))),
            None => 0.
        };
        let t = Self::sufficient_stat(y);
        self.suf_log_prob(t.slice((0, 0), (2, 1))) + loc_lp + scale_lp
    }

    fn sample(&self) -> DMatrix<f64> {
        // use rand_distr::{Distribution};
        use rand::prelude::*;
        let var = self.var()[0];
        let mut samples = DMatrix::zeros(self.mu.nrows(), 1);
        for (i, _) in self.mu.iter().enumerate() {
            let n : f64 = rand::thread_rng().sample(rand_distr::StandardNormal);
            samples[(i,1)] = var * n;
        }
        samples
    }

}

impl Likelihood<U1> for Normal {

    fn mean_mle(y : DMatrixSlice<'_, f64>) -> f64 {
        let mle = y.iter().fold(0.0, |ys, y| ys + y) / (y.nrows() as f64);
        mle
    }

    fn var_mle(y : DMatrixSlice<'_, f64>) -> f64 {
        let n = y.nrows() as f64;
        y.iter().fold(0.0, |ys, y| ys + y.powf(2.) / n)
    }

}

impl RandomWalk for Normal {

    fn current<'a>(&'a self) -> Option<DVectorSlice<'a, f64>> {
        self.eta_traj.as_ref().and_then(|eta_traj| {
            Some(eta_traj.traj.column(eta_traj.pos))
        })
    }

    fn step_to<'a>(&'a mut self, new_eta : Option<DVectorSlice<'a, f64>>, _update : bool) {
        if let Some(ref mut eta_traj) = self.eta_traj {
            eta_traj.step(new_eta)
        } else {
            self.eta_traj = Some(EtaTrajectory::new(new_eta.unwrap()));
        }
    }

    fn step_by<'a>(&'a mut self, diff_eta : DVectorSlice<'a, f64>, _update : bool) {
        self.eta_traj.as_mut().unwrap().step_increment(diff_eta);
    }

    fn marginal(&self) -> Option<Sample> {
        self.eta_traj.as_ref().and_then(|eta_traj| {
            let cols : Vec<DVector<f64>> = eta_traj.traj.clone()
                .column_iter().take(eta_traj.pos).map(|col| Self::link(&col) ).collect();
            let t_cols = DMatrix::from_columns(&cols[..]);
            Some(Sample::new(t_cols))
        })
    }

}

