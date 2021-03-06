use super::*;
use serde::{Serialize, Deserialize};
use rand_distr;
use crate::fit::markov::*;
use std::default::Default;
use serde::ser::{Serializer};
use serde::de::Deserializer;
use std::fmt::{self, Display};
use anyhow;
use crate::fit::Estimator;
use crate::calc::Variate;
use std::iter::FromIterator;

pub type PoissonFactor = UnivariateFactor<Gamma>;

/// The Poisson is the exponential-family distribution
/// used as the likelihood for countable outcomes. Each realization is parametrized
/// by a rate parameter λ (λ > 0), whose natural
/// parameter transformation is its logarithm ln(λ)
///
/// # Example
///
/// ```
/// use bayes::distr::*;
///
/// let n = 1000;
/// let mut poiss = Poisson::new(n, Some(1.0));
/// let y = poiss.sample();
///
/// // Maximum likelihood estimate
/// let mle = Poisson::mle((&y).into());
///
/// // Bayesian conjugate estimate
/// let mut poiss_cond = poiss.condition(Gamma::new(1.0,1.0));
/// poiss_cond.fit(y, None);
/// let post : Gamma = poiss_cond.take_factor().unwrap();
/// assert!(post.mean()[0] - mle.mean()[0] < 1E-3);
/// ```
#[derive(Debug, Clone)]
pub struct Poisson {

    lambda : DVector<f64>,

    eta : DVector<f64>,

    factor : PoissonFactor,

    log_part : DVector<f64>,

    // eta_traj : Option<RandomWalk>,

    sampler : Vec<rand_distr::Poisson<f64>>,

    suf_lambda : Option<DMatrix<f64>>,
    
    name : Option<String>,
    
    fixed_names : Option<Vec<String>>,
    
    obs : Option<DMatrix<f64>>,
    
    // fixed_obs : Option<DMatrix<f64>>,
    
    n : usize,

    sample : HashMap<String, Either<Vec<f64>, Vec<usize>>>

}

impl Poisson {

    pub fn new(n : usize, lambda : Option<f64>) -> Self {
        if let Some(l) = lambda {
            assert!(l > 0.0);
        }
        let mut p : Poisson = Default::default();
        p.n = n;
        p.log_part = DVector::zeros(n);
        let l = DVector::from_element(n, lambda.unwrap_or(1.));
        p.set_parameter(l.rows(0, l.nrows()), false);
        p
    }

}

impl Observable for Poisson {

    fn observations(&mut self) -> &mut Option<DMatrix<f64>> {
        &mut self.obs
    }

    fn sample_size(&mut self) -> &mut usize {
        &mut self.n
    }

}

impl ExponentialFamily<U1> for Poisson
    where
        Self : Distribution
{

    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        assert!(y.ncols() == 1);
        y.column(0).map(|y| /*1. / (Bernoulli::factorial(y as usize) as f64*/ Gamma::gamma_inv(y + 1.) )
    }

    /*fn grad(&self, y : DMatrixSlice<'_, f64>, x : Option<DMatrix<f64>>) -> DVector<f64> {
        // equivalent to sum_i { yi/lambda - 1 }
        let g = y.nrows() as f64 * (y.mean() - self.lambda[0]) / self.lambda[0];
        DVector::from_element(1, g)
    }*/

    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        assert!(y.ncols() == 1);
        DMatrix::from_element(1, 1, y.column(0).iter().fold(0.0, |ys, y| ys + y ))
    }

    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {
        assert!(t.nrows() == 1 && t.ncols() == 1);
        assert!(self.eta.nrows() == 1);
        assert!(self.log_part.nrows() == 1);
        self.eta[0] * t[0] - self.log_part[0]
    }

    fn update_log_partition<'a>(&'a mut self, /*eta : DVectorSlice<'_, f64>*/ ) {
        if self.log_part.nrows() != self.eta.nrows() {
            self.log_part = DVector::zeros(self.eta.nrows());
        }
        self.log_part.iter_mut()
            .zip(self.eta.iter())
            .for_each(|(l,e)| { *l = e.exp() } );
    }

    fn log_partition<'a>(&'a self) -> &'a DVector<f64> {
        &self.log_part
    }

    /*fn update_grad(&mut self, _eta : DVectorSlice<'_, f64>) {
        unimplemented!()
    }

    fn grad(&self) -> &DVector<f64> {
        unimplemented!()
    }*/

    fn link_inverse<S>(eta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        eta.map(|e| e.exp() )
    }

    fn link<S>(theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        theta.map(|p| p.ln() )
    }

}

impl Distribution for Poisson
    where Self : Sized
{

    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, natural : bool) {
        let eta = if natural {
            p.clone_owned()
        } else {
            Self::link(&p)
        };
        self.set_natural(&mut eta.iter());
    }

    fn set_natural<'a>(&'a mut self, new_eta : &'a mut dyn Iterator<Item=&'a f64>) {
        let (eta, lambda) = (&mut self.eta, &mut self.lambda);

        // eta.iter_mut().zip(new_eta).for_each(|(old, new)| *old = *new );
        *eta = DVector::from(Vec::from_iter(new_eta.cloned()));
        *lambda = DVector::from(Vec::from_iter(new_eta.map(|e| e.ln() )));
        self.update_log_partition();
        if let Some(ref mut suf) = self.suf_lambda {
            *suf = Gamma::sufficient_stat(self.lambda.slice((0,0), (self.lambda.nrows(),1)));
        }
        self.sampler.clear();
        for l in self.lambda.iter() {
            self.sampler.push(rand_distr::Poisson::new(*l).unwrap());
        }
    }

    fn view_parameter(&self, natural : bool) -> &DVector<f64> {
        match natural {
            true => &self.eta,
            false => &self.lambda
        }
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.lambda
    }
    
    fn mode(&self) -> DVector<f64> {
        self.lambda.clone()
    }

    fn var(&self) -> DVector<f64> {
        self.lambda.clone()
    }

    fn joint_log_prob(&self, /*y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>*/ ) -> Option<f64> {
        /*// assert!(y.ncols() == 1);
        let eta = self.eta.rows(0, self.eta.nrows());
        let factor_lp = match &self.factor {
            PoissonFactor::Conjugate(g) => {
                assert!(y.ncols() == 1);
                assert!(x.is_none());
                g.log_prob(self.suf_lambda.as_ref().unwrap().slice((0,0), (1,2)), None)
            },
            PoissonFactor::Fixed(m) => {
                m.log_prob(eta.slice((0,1), (eta.nrows(), eta.ncols() - 1)), x)
            },
            PoissonFactor::Empty => 0.
        };
        eta.dot(&y.slice((0, 0), (y.nrows(), 1))) - self.log_part[0] + factor_lp*/
        super::univariate_joint_log_prob(
            self.obs.as_ref(),
            self.factor.fixed_obs(),
            &self.factor,
            &self.view_parameter(true),
            &self.log_part,
            self.suf_lambda.clone()
        )
    }

    fn sample_into(&self, mut dst : DMatrixSliceMut<'_,f64>) {
        use rand_distr::{Distribution};
        let opt_lambda = sample_canonical_factor::<Poisson,_>(self.view_fixed_values(), &self.factor, self.n);
        let lambda = opt_lambda.as_ref().unwrap_or(&self.lambda);
        for (i, _) in lambda.iter().enumerate() {
            let s : u64 = self.sampler[i].sample(&mut rand::thread_rng());
            dst[(i,0)] = s as f64;
        }
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        None
    }

    fn cov_inv(&self) -> Option<DMatrix<f64>> {
        None
    }

    fn dyn_factors(&self) -> (Option<&dyn Distribution>, Option<&dyn Distribution>) {
         (super::univariate_factor(&self.factor), None)
    }

    fn dyn_factors_mut<'a>(&'a mut self) -> (Option<&'a mut dyn Distribution>, Option<&'a mut dyn Distribution>) {
        (super::univariate_factor_mut(&mut self.factor), None)
    }

}

/*impl Posterior for Poisson {

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Posterior>, Option<&mut dyn Posterior>) {
        match &mut self.factor {
            PoissonFactor::Conjugate(ref mut b) => (Some(b as &mut dyn Posterior), None),
            PoissonFactor::Fixed(ref mut m) => (Some(m as &mut dyn Posterior), None),
            _ => (None, None)
        }
    }

    fn set_approximation(&mut self, m : MultiNormal) {
        unimplemented!()
    }

    fn approximate(&self) -> Option<&MultiNormal> {
        unimplemented!()
    }

}*/

impl Markov for Poisson {

    fn natural_mut<'a>(&'a mut self) -> DVectorSliceMut<'a, f64> {
        self.eta.column_mut(0)
    }

    fn canonical_mut<'a>(&'a mut self) -> Option<DVectorSliceMut<'a, f64>> {
        Some(self.lambda.column_mut(0))
    }

}

impl Conditional<Gamma> for Poisson {

    fn condition(mut self, g : Gamma) -> Poisson {
        self.factor = PoissonFactor::Conjugate(g);
        self.suf_lambda = Some(Gamma::sufficient_stat(self.lambda.slice((0, 0), (self.lambda.nrows(), 1))));
        // TODO update sampler vector
        self
    }

    fn view_factor(&self) -> Option<&Gamma> {
        match &self.factor {
            PoissonFactor::Conjugate(g) => Some(g),
            _ => None
        }
    }

    fn take_factor(self) -> Option<Gamma> {
        match self.factor {
            PoissonFactor::Conjugate(g) => Some(g),
            _ => None
        }
    }

    fn factor_mut(&mut self) -> Option<&mut Gamma> {
        match &mut self.factor {
            PoissonFactor::Conjugate(g) => Some(g),
            _ => None
        }
    }

}

impl Likelihood<usize> for Poisson {

    fn observe<'a>(&mut self, obs : impl IntoIterator<Item=&'a usize>) {
        let cvt_obs : Vec<f64> = obs.into_iter().map(|val| *val as f64 ).collect();
        observe_univariate_generic(self, cvt_obs.iter());
    }
    
    fn likelihood<'a>(obs : impl IntoIterator<Item=&'a usize>) -> Self {
        let mut poiss = Poisson::new(1, None);
        poiss.observe(obs);
        poiss
    }

    fn view_variables(&self) -> Option<Vec<String>> {
        self.name.as_ref().map(|name| vec![name.clone()] )
    }
    
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
    //}

    fn observe_sample(&mut self, sample : &dyn Sample, vars : &[&str]) {
        //self.obs = Some(super::observe_univariate(self.name.clone(), self.lambda.len(), self.obs.take(), sample));
        // self.n = 0;
        let mut obs = self.obs.take().unwrap_or(DMatrix::zeros(self.lambda.nrows(), 1));
        if let Some(name) = vars.get(0) {
            if let Variable::Count(col) = sample.variable(&name) {
                for (tgt, src) in obs.iter_mut().zip(col) {
                    *tgt = src as f64;
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
    
    /*fn mle(y : DMatrixSlice<'_, f64>) -> Result<Self, anyhow::Error> {
        let lambda = y.sum() as f64 / y.nrows() as f64;
        Ok(Self::new(1, Some(lambda)))
    }*/

    /*fn mean_mle(y : DMatrixSlice<'_, f64>) -> f64 {
        assert!(y.ncols() == 1);
        let mle = y.iter().fold(0.0, |ys, y| ys + y) / (y.nrows() as f64);
        mle
    }

    fn var_mle(y : DMatrixSlice<'_, f64>) -> f64 {
        Self::mean_mle(y)
    }*/

    /*fn visit_factors<F>(&mut self, f : F) where F : Fn(&mut dyn Posterior) {
        match self.factor {
            PoissonFactor::Conjugate(ref mut b) => {
                f(b);
                b.visit_post_factors(&f as &dyn Fn(&mut dyn Posterior));
            },
            PoissonFactor::Fixed(ref mut m) => {
                f(m);
                m.visit_post_factors(&f as &dyn Fn(&mut dyn Posterior));
            },
            _ => { }
        }
    }*/

    /*fn factors_mut(&mut self) -> Factors {
        match self.factor {
            BernoulliFactor::Conjugate(b) => b.aggregate_factors(Factors::new_empty()),
            BernoulliFactor::Fixed(m) => m.aggregate_factors(Factors::new_empty()),
            _ => Factors::new_empty()
        }
    }*/

    //fn factors_mut(&mut self) -> Factors {
    //    unimplemented!()
    //}

}

/*impl Predictive for Poisson {

    fn predict<'a>(&'a mut self, fixed : Option<&dyn Sample>) -> Option<&'a dyn Sample> {
        /*super::collect_fixed_if_required(self, fixed);
        let transf = |obs : &f64| -> usize { *obs as usize };
        let preds = super::try_build_generalized_predictions(self, transf)
            .map_err(|e| println!("{}", e)).unwrap();
        self.sample = preds;
        self.view_prediction()*/
        unimplemented!()
    }

    fn view_prediction<'a>(&'a self) -> Option<&'a dyn Sample> {
        if self.sample.is_empty() {
            None
        } else {
            Some(&self.sample as &dyn Sample )
        }
    }

}*/

impl<'a> Estimator<'a, &'a Gamma> for Poisson {

    type Algorithm = ();

    type Error = &'static str;

    //fn predict<'a>(&'a self, cond : Option<&'a Sample/*<'a>*/>) -> Box<dyn Sample /*<'a>*/ > {
    //    unimplemented!()
    //}
    
    /*fn take_posterior(self) -> Option<Gamma> {
        unimplemented!()
    }
    
    fn view_posterior<'a>(&'a self) -> Option<&'a Gamma> {
        unimplemented!()
    }*/
    
    fn fit(&'a mut self, algorithm : Option<()>) -> Result<&'a Gamma, &'static str> {
        // self.observe_sample(sample);
        match self.factor {
            PoissonFactor::Conjugate(ref mut gamma) => {
                let y = self.obs.clone().unwrap();
                let n = y.nrows() as f64;
                let ys = y.column(0).sum();
                let (a, b) = (gamma.view_parameter(false)[0], gamma.view_parameter(false)[1]);
                let new_param = DVector::from_column_slice(&[a + ys, b + n]);
                gamma.set_parameter(new_param.rows(0, new_param.nrows()), false);
                Ok(&(*gamma))
            },
            _ => Err("Distribution does not have a conjugate factor")
        }
    }

}

/*impl RandomWalk for Poisson {

    fn current<'a>(&'a self) -> Option<DVectorSlice<'a, f64>> {
        self.eta_traj.as_ref().and_then(|eta_traj| {
            Some(eta_traj.traj.column(eta_traj.pos))
        })
    }

    fn step_by<'a>(&'a mut self, diff_eta : DVectorSlice<'a, f64>, _update : bool) {
        self.eta_traj.as_mut().unwrap().step_increment(diff_eta);
    }

    fn step_to<'a>(&'a mut self, new_eta : Option<DVectorSlice<'a, f64>>, update : bool) {
        if let Some(ref mut eta_traj) = self.eta_traj {
            eta_traj.step(new_eta)
        } else {
            self.eta_traj = Some(EtaTrajectory::new(new_eta.unwrap()));
        }
        if update {
            self.set_parameter(new_eta.unwrap(), true);
        }
    }

    fn marginal(&self) -> Option<Sample> {
        self.eta_traj.as_ref().and_then(|eta_traj| {
            let cols : Vec<DVector<f64>> = eta_traj.traj.clone()
                .column_iter().take(eta_traj.pos)
                .map(|col| Self::link(&col) ).collect();
            let t_cols = DMatrix::from_columns(&cols[..]);
            Some(Sample::new(t_cols))
        })
    }
}*/

impl Default for Poisson {

    fn default() -> Self {
        Poisson {
            lambda : DVector::from_element(1, 0.5),
            eta : DVector::from_element(1, 0.0),
            factor : PoissonFactor::Empty,
            // eta_traj : None,
            sampler : Vec::new(),
            suf_lambda : None,
            log_part : DVector::from_element(1, (2.).ln()),
            obs : None,
            // fixed_obs : None,
            fixed_names : None,
            name : None,
            n : 0,
            sample : HashMap::new()
        }
    }

}

impl Serialize for Poisson {

    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        unimplemented!()
    }
}

impl<'de> Deserialize<'de> for Poisson {

    fn deserialize<D>(_deserializer: D) -> Result<Poisson, D::Error>
    where
        D: Deserializer<'de>,
    {
        unimplemented!()
    }
}

impl Display for Poisson {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Poiss({})", self.lambda.nrows())
    }

}

