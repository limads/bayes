use crate::prob::*;
use crate::prob::gamma::Gamma;
use rand_distr;
use rand;
use std::f64::consts::PI;
use crate::fit::markov::*;
use std::fmt::{self, Display};
use anyhow;
use crate::fit::Estimator;

type NormalFactor = UnivariateFactor<Box<Normal>>;

/// The Normal is the exponential-family distribution
/// used as the likelihood for continuous unbounded outcomes and as priors
/// for the location parameter for such outcomes. Each realization is parametrized
/// by a location parameter μ and a common scale factor σ.
///
/// Normals works both in the role of priors or likelihoods:
///
/// To represent n(mu,sigma) use: Normal::prior(0.2, 0.1);
/// to represent n(x|mu,sigma) use Normal::likelihood(&x[..]);
///
///
/// # Example
///
/// ```
/// use bayes::distr::*;
///
/// let n = 1000;
/// let mut norm = Normal::new(n, Some(0.0), Some(1.0));
/// let y = norm.sample();
///
/// // Maximum likelihood estimate
/// let mle = Normal::mle((&y).into());
///
/// // Bayesian conjugate estimate
/// let mut norm_cond = norm.condition(Normal::new(1, Some(0.0), Some(1.0)));
/// norm_cond.fit(y, None);
/// let post : Normal = norm_cond.take_factor().unwrap();
/// assert!(post.mean()[0] - mle.mean()[0] < 1E-3);
/// ```
#[derive(Debug, Clone)]
pub struct Normal {

    // Location parameter, against which location factor is
    // evaluated.
    mu : DVector<f64>,

    // Variance parameter.
    // var : Variance,

    scaled_mu : DVector<f64>,

    loc_factor : NormalFactor,

    scale_factor : Option<Gamma>,

    // See the documentation for the mix field of MultiNormal, which explains the rationale of
    // holding mixing factors in this way.
    mix : Option<Vec<Box<Normal>>>,

    log_part : DVector<f64>,

    eta_traj : Option<Trajectory>,

    /// Holds [log(1/sigma^2); 1/sigma^2], against which
    /// any scale factors (if present) are evaluated.
    prec_suff : DVector<f64>,

    /// Holds (-0.5)*1/sigma^2
    minus_half_prec : f64,

    traj : Option<Trajectory>,

    // rng : GslRng,
    
    name : Option<String>,
    
    fixed_names : Option<Vec<String>>,
    
    obs : Option<DMatrix<f64>>,
    
    // fixed_obs : Option<DMatrix<f64>>,
    
    n : usize,

    var : f64,

    sample : Option<(Vec<String>, DMatrix<f64>)>

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
    pub fn new(n : usize, loc : Option<f64>, var : Option<f64>) -> Self {
        if let Some(s) = var {
            assert!(s > 0.0, "Variance should be a strictly positive value");
        }
        let mu = DVector::from_element(n, loc.unwrap_or(0.0));
        let loc_factor = NormalFactor::Empty;
        let scale_factor = None;
        let var = var.unwrap_or(1.);
        let prec_suff = DVector::from_column_slice(&[(1. / var).ln(), 1. / var]);
        let eta_traj = None;
        let log_part = mu.map(|e| e.powf(2.) / 2. );
        let rand_seed : f64 = rand::random();
        // let rng = GslRng::new(rand_seed as u64 * 32000);
        let mut norm = Self{ 
            mu : mu.clone(), 
            scaled_mu : mu.clone(), 
            loc_factor,
            traj : None,
            // rng,
            eta_traj, 
            prec_suff : prec_suff.clone(), 
            scale_factor, 
            log_part, 
            minus_half_prec : -prec_suff[1] / 2.,
            name : None,
            fixed_names : None,
            obs : None,
            // fixed_obs : None,
            n,
            sample : None,
            var,
            mix : None
        };
        norm.set_var(var);
        norm.set_parameter(mu.rows(0, mu.nrows()), true);
        norm
    }

    pub fn standardize(&self, val : f64) -> f64 {
        assert!(self.n == 1);
        val - self.mu[0] / self.stddev()
    }

    /// Evaluates the log-probability of each data sample individually, assuming
    /// this element does not have any factors.
    pub fn cond_log_prob(&self) -> Option<DVector<f64>> {
        assert!(self.loc_factor.is_empty());
        let y = self.obs.as_ref()?;
        let y_sq = y.column(0).map(|y| y.powf(2.0) );
        Some(self.scaled_mu.component_mul(&y) + self.minus_half_prec*y_sq - &self.log_part)
    }

    // Will not work for heteroscedastic distributions.
    pub fn stddev(&self) -> f64 {
        //let v = 1. / self.prec_suff[1];
        //v.sqrt()
        self.var.sqrt()
    }

    pub fn set_var(&mut self, var : f64) {
        self.var = var;
        let prec = 1. / var;
        self.prec_suff[0] = prec.ln();
        self.prec_suff[1] = prec;
        self.minus_half_prec = (-1.)*prec / 2.;
        self.scaled_mu = self.mu.scale(prec);
        // TODO remove requirement for argument to update_log_partition eta vector.
        // let unused : DVector<f64> = DVector::zeros(1);
        self.update_log_partition( /*(&unused).into()*/ );
    }

    /*pub fn mle(y : DMatrixSlice<'_, f64>) -> (f64, f64) {
        assert!(y.ncols() == 1);
        let n = y.nrows() as f64;
        y.iter().fold((0.0, 0.0), |ys, y| (ys.0 + y / n, ys.1 + y.powf(2.) / n) )
    }*/

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

impl Distribution for Normal
    where Self : Sized
{

    fn set_natural<'a>(&'a mut self, eta : &'a mut dyn Iterator<Item=&'a f64>) {
        self.mu.iter_mut().zip(eta).for_each(|(old, new)| *old = *new );
        self.scaled_mu = self.prec_suff[1] * self.mu.clone();
        self.update_log_partition();
    }

    // The parameter set at the eta scale should be assumed to be the scaled mu;
    // the parameter set at the theta scale should be assumed to be mu (conditional
    // on the current inverse scale).
    fn set_parameter(&mut self, mu : DVectorSlice<'_, f64>, natural : bool) {
        assert!(natural);
        self.set_natural(&mut mu.iter());
    }

    fn view_parameter(&self, _natural : bool) -> &DVector<f64> {
        // see mu; vs. see sigma_inv*mu
        &self.mu
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.mu
    }

    fn mode(&self) -> DVector<f64> {
        self.mu.clone()
    }

    fn var(&self) -> DVector<f64> {
        // let v = 1. / self.prec_suff[1];
        // DVector::from_element(1, v)
        DVector::from_element(1, self.var)
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        None
    }

    fn cov_inv(&self) -> Option<DMatrix<f64>> {
        None
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

        use rand::prelude::*;
        let var = self.var()[0];
        let sd = var.sqrt();
        for (i, m) in mu.iter().enumerate() {
            let n : f64 = rand::thread_rng().sample(rand_distr::StandardNormal);
            dst[(i,0)] = sd * (n + m);
        }
    }

    fn dyn_factors(&self) -> (Option<&dyn Distribution>, Option<&dyn Distribution>) {
        let loc = match self.loc_factor {
            NormalFactor::Conjugate(ref n) => Some(n.as_ref() as &dyn Distribution),
            NormalFactor::Fixed(ref mn) => Some(mn as &dyn Distribution),
            NormalFactor::Empty => None
        };
        let scale = self.scale_factor.as_ref().map(|sf| sf as &dyn Distribution);
        (loc, scale)
    }

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Distribution>, Option<&mut dyn Distribution>) {
        let loc = match self.loc_factor {
            NormalFactor::Conjugate(ref mut n) => Some(n.as_mut() as &mut dyn Distribution),
            NormalFactor::Fixed(ref mut mn) => Some(mn as &mut dyn Distribution),
            NormalFactor::Empty => None
        };
        let scale = self.scale_factor.as_mut().map(|sf| sf as &mut dyn Distribution);
        (loc, scale)
    }

}

impl Markov for Normal {

    fn natural_mut<'a>(&'a mut self) -> DVectorSliceMut<'a, f64> {
        self.mu.column_mut(0)
    }

    fn canonical_mut<'a>(&'a mut self) -> Option<DVectorSliceMut<'a, f64>> {
        None
    }

}

impl Posterior for Normal {

    fn approximation_mut(&mut self) -> Option<&mut MultiNormal> {
        //Some(self)
        unimplemented!()
    }

    fn approximation(&self) -> Option<&MultiNormal> {
        //Some(self)
        unimplemented!()
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

impl Likelihood<f64> for Normal {

    fn view_variables(&self) -> Option<Vec<String>> {
        self.name.as_ref().map(|name| vec![name.clone()] )
    }
    
    fn likelihood<'a>(obs : impl IntoIterator<Item=&'a f64>) -> Self {
        let mut norm = Normal::new(1, None, None);
        norm.observe(obs);
        norm
    }

    fn observe<'a>(&mut self, obs : impl IntoIterator<Item=&'a f64>) {
        observe_univariate_generic(self, obs.into_iter());
    }

    /*fn factors_mut<'a>(&'a mut self) -> (Option<&'a mut dyn Posterior>, Option<&'a mut dyn Posterior>) {
        //Conditional::<Normal>::dyn_factors_mut(&mut self)
        unimplemented!()
    }*/
    
    fn with_variables(&mut self, vars : &[&str]) -> &mut Self {
        assert!(vars.len() == 1);
        self.name = Some(vars[0].to_string());
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

    // fn view_fixed_values(&self) -> Option<&DMatrix<f64>> {
    //    self.fixed_obs.as_ref()
    // }

    fn observe_sample(&mut self, sample : &dyn Sample, vars : &[&str]) {
        //self.obs = Some(super::observe_univariate(self.name.clone(), self.mu.len(), self.obs.take(), sample));
        // self.n = 0;
        let mut obs = self.obs.take().unwrap_or(DMatrix::zeros(self.mu.nrows(), 1));
        if let Some(name) = vars.get(0) {
            if let Variable::Real(col) = sample.variable(&name) {
                for (tgt, src) in obs.iter_mut().zip(col) {
                    *tgt = src;
                    // self.n += 1;
                }
            }
        }
        self.obs = Some(obs);
        
        /*if let Some(fixed_names) = &self.fixed_names {
            let fix_names = fixed_names.clone();
            super::observe_real_columns(&fix_names[..], sample, &mut self.fixed_obs, self.n);
        }*/
        
        // self
    }
    
    /*fn mean_mle(y : DMatrixSlice<'_, f64>) -> f64 {
        let mle = y.iter().fold(0.0, |ys, y| ys + y) / (y.nrows() as f64);
        mle
    }

    fn var_mle(y : DMatrixSlice<'_, f64>) -> f64 {
        let n = y.nrows() as f64;
        y.iter().fold(0.0, |ys, y| ys + y.powf(2.) / n)
    }*/

    /*/// Biased maximum likelihood mean estimate
    fn mle(y : DMatrixSlice<'_, f64>) -> Result<Self, anyhow::Error> {
        let suff = Self::sufficient_stat(y);
        let mean = suff[0] / y.nrows() as f64;
        let var = suff[1] / y.nrows() as f64 - mean.powf(2.);
        if var <= 0.0 {
            return Err(anyhow::Error::msg("Variance of estimate cannot be zero"));
        }
        Ok(Normal::new(1, Some(mean), Some(var)))
    }*/

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

    //fn cond_log_prob(&self, _y : DMatrixSlice<'_, f64>) -> f64 {
    //    unimplemented!()
    //}

    /*fn factors_mut(&mut self) -> Factors {
        //self.dyn_factors_mut().into()
        unimplemented!()
    }*/

}

impl Conditional<Normal> for Normal {

    fn condition(mut self, n : Normal) -> Normal {
        assert!(n.n == 1);
        self.loc_factor = NormalFactor::Conjugate(Box::new(n));
        // TODO update sampler vector
        self
    }

    fn view_factor(&self) -> Option<&Normal> {
        match &self.loc_factor {
            NormalFactor::Conjugate(bx_norm) => Some(bx_norm.as_ref()),
            _ => None
        }
    }

    fn take_factor(self) -> Option<Normal> {
        match self.loc_factor {
            NormalFactor::Conjugate(bx_norm) => Some(*bx_norm),
            _ => None
        }
    }

    fn factor_mut(&mut self) -> Option<&mut Normal> {
        match &mut self.loc_factor {
            NormalFactor::Conjugate(bx_norm) => Some(bx_norm.as_mut()),
            _ => None
        }
    }

}

impl Conditional<MultiNormal> for Normal {

    fn condition(mut self, mn : MultiNormal) -> Normal {
        // assert!(mn.n == 1);
        self.loc_factor = NormalFactor::Fixed(mn);
        // TODO update sampler vector
        self
    }

    fn view_factor(&self) -> Option<&MultiNormal> {
        match &self.loc_factor {
            NormalFactor::Fixed(ref mn) => Some(mn),
            _ => None
        }
    }

    fn take_factor(self) -> Option<MultiNormal> {
        match self.loc_factor {
            NormalFactor::Fixed(mn) => Some(mn),
            _ => None
        }
    }

    fn factor_mut(&mut self) -> Option<&mut MultiNormal> {
        match &mut self.loc_factor {
            NormalFactor::Fixed(ref mut mn) => Some(mn),
            _ => None
        }
    }

}

impl Conditional<Gamma> for Normal {

    fn condition(mut self, g : Gamma) -> Normal {
        self.scale_factor = Some(g);
        // TODO update sampler vector
        self
    }

    fn view_factor(&self) -> Option<&Gamma> {
        match &self.scale_factor {
            Some(g) => Some(g),
            _ => None
        }
    }

    fn take_factor(self) -> Option<Gamma> {
        match self.scale_factor {
            Some(g) => Some(g),
            _ => None
        }
    }

    fn factor_mut(&mut self) -> Option<&mut Gamma> {
        match &mut self.scale_factor {
            Some(g) => Some(g),
            None => None
        }
    }

}

impl Mixture for Normal {

    fn mixture<const K : usize>(distrs : [Self; K], probs : [f64; K]) -> Self {
        let mix : Vec<_> = distrs[1..].iter().map(|distr| Box::new(distr.clone()) ).collect();
        let mut base = distrs[0].clone();
        base.mix = Some(mix);
        base
    }

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

impl Observable for Normal {

    fn observations(&mut self) -> &mut Option<DMatrix<f64>> {
        &mut self.obs
    }

    fn sample_size(&mut self) -> &mut usize {
        &mut self.n
    }

}

impl Stochastic for Normal {

    fn stochastic(order : usize, init : Option<&[f64]>) -> Self {
        if let Some(obs) = init {
            Normal::likelihood(obs.iter())
        } else {
            Normal::likelihood((0..order).map(|_| &0.0 ))
        }
    }

    fn evolve(&mut self, new : &[f64]) {
        assert!(new.len() == 1);
        let obs = self.obs.as_mut().unwrap().as_mut_slice();
        obs.copy_within(1.., 0);
        *obs.last_mut().unwrap() = new[0]
    }

    fn state(&self) -> &[f64] {
        let s = self.obs.as_ref().unwrap().as_slice();
        &s[s.len()-1..]
    }

}

impl Predictive for Normal {

    fn predict<'a>(&'a mut self, fixed : Option<&dyn Sample>) -> Option<&'a dyn Sample> {
        /*super::collect_fixed_if_required(self, fixed);
        let preds = super::try_build_real_predictions(self).map_err(|e| println!("{}", e)).unwrap();
        self.sample = Some(preds);
        self.view_prediction()*/
        unimplemented!()
    }

    fn view_prediction<'a>(&'a self) -> Option<&'a dyn Sample> {
        self.sample.as_ref().map(|s| s as &dyn Sample )
    }

}

/*impl Trajectory for Normal {

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

    fn marginal(&self) -> Option<Marginal> {
        self.eta_traj.as_ref().and_then(|eta_traj| {
            let cols : Vec<DVector<f64>> = eta_traj.traj.clone()
                .column_iter().take(eta_traj.pos).map(|col| Self::link(&col) ).collect();
            let t_cols = DMatrix::from_columns(&cols[..]);
            Some(Sample::new(t_cols))
        })
    }
}*/

pub mod utils {

    pub fn univariate_mle(data : &[f64]) -> (f64, f64) {
        let (sum, sum_sq) = data.iter()
            .fold((0.0, 0.0), |accum, d| {
                (accum.0 + d, accum.1 + d.powf(2.))
            });
        let n = data.len() as f64;
        let mean = sum / (n as f64);
        (mean, sum_sq / (n as f64) - mean.powf(2.))
    }

}

impl Display for Normal {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Norm({}, {})", self.mean()[0], self.var()[0] )
    }

}

// TODO implement Into<MultiNormal> (useful for EM algorithm).


