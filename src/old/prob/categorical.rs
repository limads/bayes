use nalgebra::*;
use super::*;
use super::dirichlet::*;
use serde::{Serialize, Deserialize};
use std::fmt::{self, Display};

/// Any discrete joint distribution graph with factors linked by conditional probability tables
/// resolves to a graph of categorical distributions, parametrized by a CPD.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Categorical {

    // Categorical uses the one-against-k parametrization. In this case, there is a "default"
    // class that receive probability 1 - sum(theta); the remaining classes receive probabilities
    // according to their entries at theta.
    theta : DVector<f64>,

    log_theta : DVector<f64>,

    eta : DVector<f64>,

    // Unlike the univariate distributions, that hold a log-partition vector with same dimensionality
    // as the parameter vector, the categorical holds a parameter vector with a single value, representing
    // the state of the theta vector.
    log_part : DVector<f64>,

    // factor : Option<Dirichlet>,

    obs : Option<DMatrix<f64>>,

    n : usize
}

impl Categorical {

    pub fn new(n : usize, k : usize, param : Option<&[f64]>) -> Self {
        //println!("{}", param);
        //println!("{}", param.sum() + (1. - param.sum()) );
        let param = match param {
            Some(p) => {
                assert!(p.len() == k - 1);
                DVector::from_column_slice(p)
            },

            // Attribute same probability at one-against-k parametrization
            None => DVector::from_element(k, 1. / ((k + 1) as f64))

            // One-in-k parametrization
            // None => DVector::from_element(k, 1. / k as f64)
        };
        assert!(param.sum() + (1. - param.sum()) - 1. < 10E-8);
        let eta = Self::link(&param);
        let mut cat = Self {
            log_theta : param.map(|t| t.ln() ),
            theta : param,
            eta : eta.clone(),
            log_part : DVector::from_element(1,1.),
            // factor : None,
            n,
            obs : None
        };
        cat.update_log_partition();
        cat
    }

    /*// Resolves the conditional log-probability for each realization, assuming
    // the one-in-k parametrization.
    fn cond_log_prob(&self) -> Option<DVector<f64>> {

        if let Some(obs) = self.obs.as_ref() {
            let n = obs.nrows();
            let mut probs = DVector::zeros(n);
            for row in obs.row_iter() {
                probs[i] = if row[]
            }
        } else {
            None
        }
    }*/

    fn prob_for_class(&self, class : usize) -> f64 {
        self.theta[class]
    }

}

impl Distribution for Categorical {

    fn sample(&self, dst : &mut [f64]) {
        unimplemented!()
    }

    fn view_parameter(&self, natural : bool) -> &DVector<f64> {
        match natural {
            true => &self.eta,
            false => &self.theta
        }
    }

    fn set_natural<'a>(&'a mut self, new_eta : &'a mut dyn Iterator<Item=&'a f64>) {
        let (eta, theta) = (&mut self.eta, &mut self.theta);
        eta.iter_mut().zip(new_eta).for_each(|(old, new)| *old = *new );
        self.theta = Self::link_inverse(&self.eta);
        self.log_theta = self.theta.map(|t| t.ln());
        self.update_log_partition();
    }

    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, natural : bool) {
        if natural {
            self.set_natural(&mut p.iter());
        } else {
            let theta = p.clone_owned();
            let eta = Self::link(&theta);
            self.set_natural(&mut eta.iter());
        }
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.theta
    }

    fn mode(&self) -> DVector<f64> {
        self.theta.clone()
    }

    fn var(&self) -> DVector<f64> {
        self.theta.map(|theta| theta * (1. - theta))
    }

    fn joint_log_prob(&self, /*y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>*/ ) -> Option<f64> {
        /*let t  = Self::sufficient_stat(y.rows(0, y.nrows()));
        let factor_lp = match &self.factor {
            Some(dir) => {
                dir.suf_log_prob(self.log_theta.slice((0, 0), (self.log_theta.nrows(), 1)))
            },
            None => 0.0
        };
        self.suf_log_prob(t.rows(0, t.nrows())) + factor_lp*/
        unimplemented!()
    }

    fn sample_into(&self, _dst : DMatrixSliceMut<'_, f64>) {
        unimplemented!()
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        let d = self.mean().nrows();
        let mut m = DMatrix::zeros(d, d);
        m.set_diagonal(&DVector::from_element(d, 1.));
        Some(m)
    }

    fn cov_inv(&self) -> Option<DMatrix<f64>> {
        self.cov()
    }

}

impl Observable for Categorical {

    fn observations(&mut self) -> &mut Option<DMatrix<f64>> {
        &mut self.obs
    }

    fn sample_size(&mut self) -> &mut usize {
        &mut self.n
    }

}

#[test]
fn categorical() {

}

/*pub struct Category<N> {

}

impl<N> Category<N> {

}*/

/*impl Likelihood<usize> for Categorical {

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

}*/

/*impl Markov for Categorical {

    fn natural_mut<'a>(&'a mut self) -> DVectorSliceMut<'a, f64> {
        self.eta.column_mut(0)
    }

    fn canonical_mut<'a>(&'a mut self) -> Option<DVectorSliceMut<'a, f64>> {
        Some(self.theta.column_mut(0))
    }

}*/

impl ExponentialFamily<Dynamic> for Categorical
    where
        Self : Distribution
{

    fn base_measure(_y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        DVector::from_element(1,1.)
    }

    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        DMatrix::from_column_slice(y.ncols(), 1, y.row_sum().as_slice())
    }

    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {
        assert!(t.ncols() == 1);
        self.eta.dot(&t.column(0)) - self.log_part[0]
    }

    // Log-partition as a function of eta
    fn update_log_partition<'a>(&'a mut self) {
        let e_sum = self.eta.iter().fold(0.0, |acc, e| acc + e.exp() );
        self.log_part[0] = (1. + e_sum).ln();
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
        let exp_eta = eta.map(|e| e.exp());
        let exp_sum = DVector::from_element(eta.nrows(), exp_eta.sum());
        let exp_sum_inv = exp_sum.map(|s| 1. / (1. + s));
        exp_sum_inv.component_mul(&exp_eta)
    }

    // The categorical link is the log-odds of each non-default outcome against
    // the default outcome.
    fn link<S>(theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        let prob_compl = 1. - theta.sum();
        theta.map(|t| (t / prob_compl).ln() )
    }

}

#[test]
fn cat_test() {
    let theta = DVector::from_vec(vec![0.25, 0.25, 0.25]);
    let eta = Categorical::link(&theta);
    let theta = Categorical::link_inverse(&eta);
    println!("{}; {}", eta, theta);
}

impl Latent for Categorical {

    fn latent(n : usize) -> Self {
        // When implementing Condition<Categorical> for multinormal, resize
        // the inner vector to accomodate how many distinct conditionals there are.
        Categorical::new(n, 1, None)
    }

}
/*impl Conditional<Dirichlet> for Categorical {

    fn condition(self, _d : Dirichlet) -> Self {
        unimplemented!()
    }

    fn view_factor(&self) -> Option<&Dirichlet> {
        unimplemented!()
    }

    fn take_factor(self) -> Option<Dirichlet> {
        unimplemented!()
    }

    fn factor_mut(&mut self) -> Option<&mut Dirichlet> {
        unimplemented!()
    }

}*/

impl Display for Categorical {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cat({})", self.theta.nrows())
    }

}

// fn dist_matrix()

// pub struct Cluster {
//     center : DVector<f64>
// }


