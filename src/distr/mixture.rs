use nalgebra::*;
use super::*;
use super::categorical::*;
use super::multinormal::*;
use super::dirichlet::*;

/// A mixture is defined by a linear combination of normal probability
/// distributions whose weights result from a categorical distribution draw,
/// which is a natural way to model a discrete
/// mutually-exclusive process affecting an independent continuous outcome:
/// p(y) = prod( p_i p(y_i)  )
/// This is essentially a marginalized distribution, where the p_i
/// are the discrete marginalization factors. The marginal is what is observed
/// at a random process (suppose we have k clusters, but we do not know
/// their centroids or dispersion); in this situation this representation essentially lets
/// us evaluate the log-probabilities of a proposed centroid vector and dispersion matrices
/// at all alternative outcomes, which are marginalied at the fixed values of the latent discrete
/// variable.
/// Since the inner probability does not factor with the rest of the log-
/// probabilities in the graph, the received outcome should be compared against
/// all possible combinations of the discrete outcome before being propagated
/// back into the graph.
/// This operation can be expressed as a product between a categorical
/// and a multivariate normal outcome: The dot-product between the categorical output
/// and the mean vector propagate a univariate normal distribution in the forward pass;
/// and the products of all potential realizations with the fixed values of the rhs define
/// a parameter vector to be propagated in the backward pass to both branches.
/// If we use this mixture and take the product again, the categorical can be interpreted
/// as being the LHS of the dot product with the row-stacked multivariate means, in which
/// case the mixture is selecting one of k possible multivariate realizations.
pub struct NormalMixture {

    /// Underlying vector of means, against which the vector or matrix of
    /// log-probabilities is compared against the RHS.
    normals : Vec<MultiNormal>,

    /// Underlying fixed categorical, against which the vector or matrix of
    /// log-probabilities is compared against the LHS.
    cat : Categorical,

    /// Matrix of all possible categorical realizations, arranged over rows. The matrix product
    /// of this matrix with a column vector RHS yields a vector of univariate normal
    /// realizations, against which the log-probability of normals::Univariate can be masured;
    /// The matrix product of this matrix with a matrix of RHS multivariate normal means over rows
    /// yields a matrix of multivariate realizations, each row being a set of candidate class
    /// log-probabilities against which the log-probability of normals::Multivariate can be measured.
    /// The log-probability is calculated by using the underlying categorical as a set of weights
    /// for the realized values.
    realiz : DMatrix<f64>,

    /// Matrix of realizations; transposed.
    realiz_trans : DMatrix<f64>

}

impl NormalMixture {

    pub fn new(normals) -> Self {
        unimplemented!()
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

    fn sample(&self) -> DMatrix<f64> {
        unimplemented!()
    }

}

