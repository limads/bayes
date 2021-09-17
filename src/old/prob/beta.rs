use super::*;
use super::gamma::*;
use rand_distr;
use rand;
use std::default::Default;
use std::fmt::{self, Display};
use super::MultiNormal;
use crate::fit::markov::Trajectory;
use std::convert::TryFrom;
use serde_json::{self, Value, Number};

/// A beta distribution yields ratios over the interval [0, 1], produced by taking the
/// ratio of two independent gamma distributions: If u ~ Gamma(n/2, 0.5) and v ~ Gamma(m/2, 0.5)
/// Then u / (u + v) ~ Beta. It is commonly used to model prior Bernoulli probabilities, where
/// m and n are the pseudo-data dimensionality of a success trial count and its complement.
/// By setting u = v >> 0, Beta approaches a gaussian distribution centered at u / (u + v). By setting
/// u = v = 1, Beta equals a uniform distribution bounded at [0,1]. Alternatively, a Beta can be seen
/// as a closed-form, finite-sample, slightly biased distribution for the estimate of a proportion parameter,
/// where the bias is given by the size of the pseudo-sample considered.
#[derive(Debug, Clone)]
pub struct Beta {

    // For a full joint evaluation, what its the first index and the lenght of the parameters
    // that correspond to this distribution.
    // joint_ix : (usize, usize),

    // Represent vector of a; vector of b.
    ab : DVector<f64>,

    mean : DVector<f64>,

    sampler : rand_distr::Beta<f64>,

    log_part : DVector<f64>,

    factor : Option<Box<Beta>>,

    traj : Option<Trajectory>,

    approx : Option<MultiNormal>

}

// use rand_distr::{Distribution, Beta};
// let beta = Beta::new(2.0, 5.0).unwrap();
// let v = beta.sample(&mut rand::thread_rng());
// println!("{} is from a Beta(2, 5) distribution", v);

impl Beta {

    pub fn new(a : usize, b : usize) -> Self {
        assert!(a > 0);
        assert!(b > 0);
        let mut beta : Beta = Default::default();
        let ab = DVector::from_column_slice(&[a as f64, b as f64]);
        beta.set_parameter(ab.rows(0, 2), false);
        beta
    }

}

impl ExponentialFamily<Dynamic> for Beta {

    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        //println!("y={}", y);
        if y.ncols() > 2 {
            panic!("The Beta distribution can only be evaluated at a single data point");
        }
        let theta = y[0];
        let theta_trunc = if theta == 0.0 {
            1E-10
        } else {
            if theta == 1. {
                1. - 1E-10
            } else {
                theta
            }
        };
        DVector::from_element(1, 1. / (theta_trunc * (1. - theta_trunc)) )
    }

    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        assert!(y.ncols() == 1, "Beta should be evaluated against a single column sample");
        let mut suf = DMatrix::zeros(2, 1);
        for y in y.column(0).iter() {
            let y_trunc = if *y == 0.0 {
                1E-10
            } else {
                if *y == 1. {
                    1. - 1E-10
                } else {
                    *y
                }
            };
            suf[(0,0)] += y_trunc.ln();
            suf[(1,0)] += (1. - y_trunc).ln()
        }
        suf
    }

    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {
        assert!(t.ncols() == 1 && t.nrows() == 2, "Sufficient probability matrix of beta should be 2x1");
        assert!(self.log_part.nrows() == 1, "Sufficient probability matrix of beta should be 2x1");
        self.ab.dot(&t.column(0)) - self.log_part[0]
    }

    fn update_log_partition<'a>(&'a mut self /*, eta : DVectorSlice<'_, f64>*/ ) {
        //println!("{}", eta);
        let log_part_val = Gamma::ln_gamma(self.ab[0] as f64) +
            Gamma::ln_gamma(self.ab[1] as f64) -
            Gamma::ln_gamma(self.ab[0] as f64 + self.ab[1] as f64 );
        self.log_part = DVector::from_element(1, log_part_val);
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
        DVector::from_iterator(eta.nrows(), eta.iter().map(|t| *t))
    }

    fn link<S>(theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
       DVector::from_iterator(theta.nrows(), theta.iter().map(|t| *t))
    }
}

impl Distribution for Beta
    where Self : Sized
{

    fn sample(&self, dst : &mut [f64]) {
        use rand_distr::Distribution;
        for i in 0..dst.len() {
            let b = self.sampler.sample(&mut rand::thread_rng());
            dst[i] = b;
        }
    }

    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, natural : bool) {
        assert!(natural);
        self.set_natural(&mut p.iter());
    }

    fn set_natural<'a>(&'a mut self, eta : &'a mut dyn Iterator<Item=&'a f64>) {
        self.ab[0] = *eta.next().unwrap();
        self.ab[1] = *eta.next().unwrap();
        let (a, b) = (self.ab[0], self.ab[1]);
        self.mean = DVector::from_element(1, a / (a + b));
        self.update_log_partition();
    }

    fn dyn_factors(&self) -> (Option<&dyn Distribution>, Option<&dyn Distribution>) {
        match &self.factor {
            Some(ref b) => (Some(b.as_ref() as &dyn Distribution), None),
            None => (None, None)
        }
    }

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Distribution>, Option<&mut dyn Distribution>) {
        match &mut self.factor {
            Some(ref mut b) => (Some(b.as_mut() as &mut dyn Distribution), None),
            None => (None, None)
        }
    }

    fn view_parameter(&self, _natural : bool) -> &DVector<f64> {
        &self.ab
    }
    
    fn cov(&self) -> Option<DMatrix<f64>> {
        None
    }

    fn cov_inv(&self) -> Option<DMatrix<f64>> {
        None
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.mean
    }

    fn mode(&self) -> DVector<f64> {
        let (a, b) = (self.ab[0], self.ab[1]);
        DVector::from_column_slice(&[a - 1., a + b - 2.])
    }

    fn var(&self) -> DVector<f64> {
        let (a, b) = (self.ab[0], self.ab[1]);
        DVector::from_element(1, a*b / (a + b).powf(2.) * (a + b + 1.))
    }

    fn joint_log_prob(&self /*, y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>*/ ) -> Option<f64> {
        /*assert!(y.ncols() == 1);
        let t = Beta::sufficient_stat(y);
        self.suf_log_prob((&t).into())*/
        unimplemented!()
    }

    /*fn shared_factors(&self) -> Factors {
        Vec::new().into()
    }

    /*fn cond_log_prob(&self, y : DMatrixSlice<f64>, joint : &DVector<f64>) -> f64 {
        unimplemented!()
    }*/

    /*fn update(&mut self, joint : &DVector<f64>) {
        unimplemented!()
    }*/*/

    fn sample_into(&self, mut dst : DMatrixSliceMut<'_,f64>) {
        use rand_distr::Distribution;
        let b = self.sampler.sample(&mut rand::thread_rng());
        dst[(0,0)] = b;
    }

    /*fn factors<'b>(&'b self) -> Factors<'b> {
        unimplemented!()
    }

    /// Iterate over mutable references of the nodes of this
    /// distribution, so they can be conditioned at other values. If this distribution
    /// is a root node, this iterator yield no values.
    fn factors_mut<'b>(&'b mut self) -> FactorsMut<'b> {
        unimplemented!()
    }*/

    /*fn marginalize(&self) -> Option<Histogram> {
        None
    }*/

    // fn retrieve_parameter<'a>(&'a self, joint : &'a DVector<f64>) -> Option<DVectorSlice<'a, f64>> {
        //Some(joint.rows(self.joint_ix.0, self.joint_ix.1))
    //    unimplemented!()
    // }

}

/*impl Markov for Beta {

    fn natural_mut<'a>(&'a mut self) -> DVectorSliceMut<'a, f64> {
        self.ab.column_mut(0)
    }

    fn canonical_mut<'a>(&'a mut self) -> Option<DVectorSliceMut<'a, f64>> {
        None
    }

}*/

/*impl Posterior for Beta {

    fn approximation_mut(&mut self) -> Option<&mut MultiNormal> {
        self.approx.as_mut()
    }

    fn approximation(&self) -> Option<&MultiNormal> {
        self.approx.as_ref()
    }

    fn start_trajectory(&mut self, size : usize) {
        self.traj = Some(Trajectory::new(size, self.view_parameter(true).nrows()));
    }
    
    /// Finish the trajectory before its predicted end.
    fn finish_trajectory(&mut self) {
        self.traj.as_mut().unwrap().closed = true;
    }
    
    fn trajectory(&self) -> Option<&Trajectory> {
        self.traj.as_ref()
    }

    fn trajectory_mut(&mut self) -> Option<&mut Trajectory> {
        self.traj.as_mut()
    }

}*/

impl Default for Beta {

    fn default() -> Self {
        Self {
            ab : DVector::from_column_slice(&[1., 1.]),
            mean : DVector::from_element(1, 0.5),
            log_part : DVector::from_element(1, 0.0),
            sampler : rand_distr::Beta::new(1., 1.).unwrap(),
            factor : None,
            traj : None,
            approx : None
        }
    }

}

impl Display for Beta {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Beta(1)")
    }

}

impl TryFrom<serde_json::Value> for Beta {

    type Error = String;

    fn try_from(val : Value) -> Result<Self, String> {
        match val.get("a") {
            Some(Value::Number(na)) => if let Some(a) = na.as_u64() {
                match val.get("b") {
                    Some(Value::Number(nb)) => if let Some(b) = nb.as_u64() {
                        Ok(Self::new(a as usize, b as usize))
                    } else {
                        Err(format!("Could not parse 'b' parameter as usize"))
                    },
                    _ => Err(format!("No valid 'b' parameter found"))
                }
            } else {
                Err(format!("Could not parse 'a' parameter as usize"))
            },
            _ => Err(format!("No valid a parameter found"))
        }
    }

}

impl Into<serde_json::Value> for Beta {

    fn into(mut self) -> serde_json::Value {
        let mut child = serde_json::Map::new();
        /*if let Some(mut obs) = self.obs.take() {
            let obs_vec : Vec<f64> = obs.data.into();
            let obs_value : Value = obs_vec.into();
            child.insert(String::from("obs"), obs_value);
        }*/
        child.insert(String::from("a"), Value::Number(Number::from_f64(self.ab[0]).unwrap()));
        child.insert(String::from("b"), Value::Number(Number::from_f64(self.ab[1]).unwrap()));
        Value::Object(child)
    }

}

/*impl ConditionalDistribution<BinaryOp<F,G>> for Beta {
    unimplemented!()
}*/
