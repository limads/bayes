// use nalgebra::*;
use super::{Distribution, Likelihood, Conditional, Prior, Posterior};
// use super::categorical::Categorical;
// use super::multinormal::*;
use crate::prob::Categorical;
use std::fmt::{self, Display};
// use serde::{Serialize, Deserialize};
// use std::rc::Rc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Mixture {

    /// Underlying vector of means, against which the vector or matrix of
    /// log-probabilities is compared against the RHS.
    mns : Vec<Box<MultiNormal>>,

    /// Underlying fixed categorical, against which the vector or matrix of
    /// log-probabilities is compared against the LHS.
    cat : crate::prob::Categorical,

    curr : usize

    /*/// Matrix of all possible categorical realizations, arranged over rows. The matrix product
    /// of this matrix with a column vector RHS yields a vector of univariate normal
    /// realizations, against which the log-probability of normals::Univariate can be masured;
    /// The matrix product of this matrix with a matrix of RHS multivariate normal means over rows
    /// yields a matrix of multivariate realizations, each row being a set of candidate class
    /// log-probabilities against which the log-probability of normals::Multivariate can be measured.
    /// The log-probability is calculated by using the underlying categorical as a set of weights
    /// for the realized values.
    realiz : DMatrix<f64>,

    /// Matrix of realizations; transposed.
    realiz_trans : DMatrix<f64>*/

}

impl Mixture {

    pub fn new(base : MultiNormal, n : usize) -> Self {
        let mns : Vec<Box<MultiNormal>> = (0..n).map(|_| Box::new(base.clone()) ).collect();
        let cat = Categorical::new(mns.len(), None);
        Mixture {
            mns,
            cat,
            curr : 0
        }
    }

}

/// Prior for the categorical distribution. Any priors for the MultiNormals
/// are assumed to be in place when the multinormals are aggregated into
/// the mixture.
impl ConditionalDistribution<Dirichlet> for NormalMixture {

    fn condition(mut self, mut d : Dirichlet) -> Bernoulli {
        self.cat.factor = Factor::Conjugate(Box::new(d));
        self
    }

    fn take_factor(self) -> Option<Dirichlet> {
        match self.cat.factor {
            Factor::Conjugate(d) => {
                Some(*d)
            },
            _ => None
        }
    }

    fn share_factor(&mut self, d : NormalMixture) {
        unimplemented!()
    }

}

impl Distribution for NormalMixture
    where Self : Debug
{

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        unimplemented!()
    }

    fn set_natural<'a>(&'a mut self, eta : &'a mut dyn Iterator<Item=&'a f64>) {
        unimplemented!()
    }

    fn set_parameter(&mut self, p : &[f64]) {
        unimplemented!()
    }

    fn mode(&self) -> DVector<f64> {
        unimplemented!()
    }

    fn var(&self) -> DVector<f64> {
        unimplemented!()
    }

    fn factors<'a>(&'a self) -> Factors {
        unimplemented!()
    }

    fn factors_mut<'a>(&'a mut self) -> FactorsMut {
        unimplemented!()
    }

    fn shared_factors<'a>(&'a self) -> Factors {
        unimplemented!()
    }

    fn marginalize(&self) -> Option<Histogram> {
        None
    }

    fn log_prob(&self, y : DMatrixSlice<f64>) -> f64 {
        let n = y.ncols();
        let p = self.normals.len();
        let means : Vec<_> = self.normals.map(|n| n.mean()).collect();
        let mean_mat = DMatrix::from_columns(means);
        let mix_avgs = self.realizations * mean_mat;
        let mut lp = 0.0;
        for i in 0..n {
            lp += self.normals[i].log_prob(mix_avgs.column(i)) +
                self.cat.log_prob(self.realiz_trans.column(i));
        }
        lp
    }

    fn sample_into(&self, dst : DMatrixSliceMut<'_, f64>) {
        unimplemented!()
    }

}

impl Posterior for NormalMixture {

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Posterior>, Option<&mut dyn Posterior>) {
        (Some(&mut self.cat as &mut dyn Posterior), None)
    }

    fn set_approximation(&mut self, m : MultiNormal) {
        unimplemented!()
    }

    fn approximate(&self) -> Option<&MultiNormal> {
        unimplemented!()
    }

}

impl Display for NormalMixture {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mix({})", self.cat.mean().nrows())
    }

}*/

