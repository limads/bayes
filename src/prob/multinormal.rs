use nalgebra::Vector;

// This should implement Exponential.
pub struct Marginal<'a, D> {
    ix : usize,
    d : &'a D
}

pub struct Bivariate<'a, D> {
    ix : usize,
    d : &'a D
}

impl<'a> Marginal<'a, MultiNormal> {

    pub fn index(&self) -> usize {
        self.ix
    }

    /// Returns a slice with the corresponding row of the covariance matrix
    pub fn cov(&self) -> f64 {
        unimplemented!()
    }
}

impl<'a> Bivariate<'a, MultiNormal> {

    pub fn index(&self) -> usize {
        self.ix
    }

    /// Returns a slice with the corresponding row of the covariance matrix
    pub fn cov(&self) -> f64 {
        unimplemented!()
    }

    pub fn corr(&self) -> f64 {
        unimplemented!()
    }

}

/// The MultiNormal **conditioned** on a covariance satisfies exponential (because it is
/// parametrized by a vector); The MultiNormal with random covariance does not.
/// TODO make const N : usize
/// A MVN can be represented either with a directed graph where each node is a normal conditional distribution,
/// with its mean being a linear combination of the parent's mean and a bias term; Or with an undirected graph, where each
/// node has a marginal mean but holds the symmetrical correlation with the other variables in the edge. The
/// second representation is unique; and it factors into infinitely many possible realizations of the first
/// representation.
/// To build an undirected graph: a.joint(b, 0.2).joint(c, 0.3)
/// To build a directed graph: a.condition([b, c], [10.0, 2.0])
/// The directed representation is a consequence of the expression for conditioning a full joint MVN.
#[derive(Debug)]
pub struct MultiNormal {

}

pub struct Component;

impl Component {

    pub fn vector(&self) -> &[f64] {
        unimplemented!()
    }

    pub fn value(&self) -> f64 {
        unimplemented!()
    }

}

pub struct Discriminant;

impl Discriminant {

    pub fn vector(&self) -> &[f64] {
        unimplemented!()
    }

    pub fn value(&self) -> f64 {
        unimplemented!()
    }

}

impl MultiNormal {

    // Iterate over the orthogonal principal components of this multinormal
    // pub fn components() -> impl Iterator<Item=Component>

    // Iterate over the ortogonal discriminant axes between two multinormals
    // pub fn discrimiants(other : &MultiNormal) -> impl Iterator<Item=Discriminant>

    /*pub fn marginals<'a>(&'a self) -> impl Iterator<Item=Marginal<'a, Self>> {
        //Vec::new().iter()
        unimplemented!()
    }*/

    // The returned iterator has size N/2 + n, corresponding to the row-wise
    // iteration over the upper triangular portion of the covariance matrix.
    /*pub fn bivariates<'a>(&'a self) -> impl Iterator<Item=Bivariate<'a, Self>> {
        // Vec::new().iter()
        unimplemented!()
    }

    pub fn bivariate<'a>(&'a self, ix_a : usize, ix_b : usize) -> Option<Bivariate<'a, Self>> {
        //self.marginals().nth(ix)
        unimplemented!()
    }

    pub fn marginal<'a>(&'a self, ix : usize) -> Option<Marginal<'a, Self>> {
        // self.marginals().nth(ix)
        unimplemented!()
    }*/

    /*/// Since covariance is positive definite, we can just slice the columns
    /// of the matrix, treating them as if they were the rows. Also export
    /// cov_mat, which returns the full covariance matrix object.
    pub fn cov_slices(&self) -> &[&[f64]] {
        unimplemented!()
    }

    pub fn cov(&self, ix : usize) -> &[f64] {
        self.cov_slices()[ix]
    }*/

}

/// # panics
/// Panics if the size of the informed buffer is different than
/// the size of the desired multinormal.
impl<const N : usize> rand_distr::Distribution<[f64; N]> for MultiNormal {

    fn sample<R>(&self, rng: &mut R) -> [f64 ; N]
    where
        R: rand::Rng + ?Sized
    {
        // use rand::prelude::*;
        // let z : f64 = rng.sample(rand_distr::StandardNormal);
        // self.scale.sqrt() * (z + self.loc)
        unimplemented!()
    }

}


