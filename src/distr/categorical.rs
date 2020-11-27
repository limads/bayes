use nalgebra::*;
use super::*;
use super::dirichlet::*;
// use std::fmt::{self, Display};
use serde::{Serialize, Deserialize};
use std::fmt::{self, Display};

/// Any discrete joint distribution graph with factors linked by conditional probability tables
/// resolves to a graph of categorical distributions, parametrized by a CPD.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Categorical {
    theta : DVector<f64>,
    log_theta : DVector<f64>,
    eta : DVector<f64>,

    /// Unlike the univariate distributions, that hold a log-partition vector with same dimensionality
    /// as the parameter vector, the categorical holds a parameter vector with a single value, representing
    /// the state of the theta vector.
    log_part : DVector<f64>,

    factor : Option<Dirichlet>
}

impl Categorical {

    pub fn new(n : usize, param : Option<&[f64]>) -> Self {
        //println!("{}", param);
        //println!("{}", param.sum() + (1. - param.sum()) );
        let param = match param {
            Some(p) => {
                assert!(p.len() == n);
                DVector::from_column_slice(p)
            },
            None => DVector::from_element(n, 1. / ((n + 1) as f64))
        };
        assert!(param.sum() + (1. - param.sum()) - 1. < 10E-8);
        let eta = Self::link(&param);
        let mut cat = Self {
            log_theta : param.map(|t| t.ln() ),
            theta : param,
            eta : eta.clone(),
            log_part : DVector::from_element(1,1.),
            factor : None
        };
        cat.update_log_partition(eta.rows(0,eta.nrows()));
        cat
    }

}

impl Distribution for Categorical {

    fn view_parameter(&self, natural : bool) -> &DVector<f64> {
        match natural {
            true => &self.eta,
            false => &self.theta
        }
    }

    fn set_parameter(&mut self, p : DVectorSlice<'_, f64>, natural : bool) {
        if natural {
            self.eta = p.clone_owned();
            self.theta = Self::link_inverse(&self.eta);
        } else {
            self.theta = p.clone_owned();
            self.eta = Self::link(&self.theta);
        }
        self.log_theta = self.theta.map(|t| t.ln());
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

    fn log_prob(&self, y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>) -> f64 {
        let t  = Self::sufficient_stat(y.rows(0, y.nrows()));
        let factor_lp = match &self.factor {
            Some(dir) => {
                dir.suf_log_prob(self.log_theta.slice((0, 0), (self.log_theta.nrows(), 1)))
            },
            None => 0.0
        };
        self.suf_log_prob(t.rows(0, t.nrows())) + factor_lp
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

    fn update_log_partition<'a>(&'a mut self, eta : DVectorSlice<'_, f64>) {
        let e_sum = eta.iter().fold(0.0, |acc, e| acc + e.exp() );
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

    fn link<S>(theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        let prob_compl = 1. - theta.sum();
        theta.map(|t| (t / prob_compl).ln() )
    }

}

impl Conditional<Dirichlet> for Categorical {

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

}

impl Display for Categorical {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cat({})", self.theta.nrows())
    }

}

// fn dist_matrix()

// pub struct Cluster {
//     center : DVector<f64>
// }

/// For an agglomerative clustering algorithm, this enum identifies
/// which distance between pairs of elements in each cluster will be used to
/// define the cluster fusion: In the single linkage strategy (Nearest neighbor), the distance
/// between the closest elements (each from one of the cluster) will be used;
/// In the complete linkage strategy (Farthest neighbor), the distance between the farthest elements (each from
/// one of the clusters) will be used.
pub enum Neighbor {
    Nearest,
    Farthest
}

/// There are two ways to build a dendrogram: (1) Divisive - Iterate over the distance matrix,
/// find the smallest distance yet unassigned to a cluster (with either both elements
/// unassigned to start a new cluster or just one element unassigned to add it to an existing cluster)
/// or (2) Agglomerative - At each iteration build all possible clusters of the same size 
/// by fusing all possible pairs of clusters using either the distance to the 
/// closest element in the cluster or the distance to the farthest element in the cluster.
/// If the user chooses the first strategy, a partial execution of the algorithm will end
/// with several un-classified elements (the ones with the largest distances); while if the user
/// chooses the second strategy, a partial execution will end with all elements classified at
/// clusters of the same size (although not necessarily the "best" ones). 
pub enum Strategy {
    Divisive,
    Agglomerative(Neighbor)
}

/// Represents the result of execuing an hierarchical clustering algorithm.
/// Useful to build priors for categorical variables when there are no class labels
/// that can be used to build a Categorical distribution. The user must choose an aggregation
/// strategy as well as the desired aggregation level. A standard way to build categorical
/// models in the absence of class labels is to fit k different categorical 
/// models at k different clustering levels in a full dendrogram (usually well-spaced), using
/// the multinormal MLEs of the elements within each cluster. The model
/// which yields the best log-likelihood is then chosen. Alternatively, the user might explore
/// the dendrogram iteratively by exploring the dendrogram areas which suggest a local maxima
/// of the categorical log-likelihood. 
pub struct Dendrogram {
    /// Outer vec: Clusters; Inner vec: cluster item indices.
    clusters : Vec<Vec<usize>>
}

/// Builds a square euclidian distance matrix from a sequence
/// of observations, each assumed to have the same dimension. 
fn distance_matrix<I,C>(data : I) -> DMatrix<f64> 
where
    I : Iterator<Item=C>,
    C : AsRef<[f64]>
{
    // Build the "wide" data matrix - Observations are columns.
    let vecs : Vec<_> = data
        .map(|item| DVector::from_iterator(
            item.as_ref().len(), 
            item.as_ref().iter().cloned()) 
        ).collect();
    let wide = DMatrix::from_columns(&vecs[..]);
    let wide_t = wide.transpose();
    
    // Build the Gram matrix (Matrix of dot products)
    // Reference: https://en.wikipedia.org/wiki/Euclidean_distance_matrix#Relation_to_Gram_matrix
    let gram = wide * wide_t;
    
    assert!(gram.nrows() == gram.ncols());
    
    // Calculate the Euclidian distance matrix from the Gram matrix
    let mut eucl = DMatrix::zeros(gram.nrows(), gram.ncols());
    for i in 0..gram.nrows() {
        for j in 0..gram.ncols() {
            eucl[(i, j)] = gram[(i, i)] - 2. * gram[(i, j)] + gram[(j, j)];
        }
    }
    
    // All diagonals must be zero (Eucl. dist. matrix is hollow)
    assert!(eucl[(0,0)] < 1E-8);
    
    eucl
}

/// When iterating over an distance matrix, we can have the row observation
/// assigned, the column observation assigned, neither, or both.
enum Assignment {
    Neither,
    Row(usize),
    Col(usize),
    Both((usize, usize))
}

impl Dendrogram {

    /// Returns which observations of the informed pair are already assigned, and at which cluster.
    fn is_assigned(clusters : &[Vec<usize>], obs_row : usize, obs_col : usize) -> Assignment {
        let mut row_assigned = None;
        let mut col_assigned = None;
        for (clust_ix, cluster) in clusters.iter().enumerate() {
            if row_assigned.is_none() && cluster.iter().find(|o| **o == obs_row ).is_some() {
                row_assigned = Some(clust_ix);
            }
            if col_assigned.is_none() && cluster.iter().find(|o| **o == obs_col ).is_some() {
                col_assigned = Some(clust_ix);
            }
        }
        match (row_assigned, col_assigned) {
            (Some(r_ix), Some(c_ix)) => Assignment::Both((r_ix, c_ix)),
            (Some(r_ix), None) => Assignment::Row(r_ix),
            (None, Some(c_ix)) => Assignment::Col(c_ix),
            (None, None) => Assignment::Neither,
        }
    }
    
    /// Builds a dendrogram using a divisive strategy, until a number of observations
    /// given by level are classified. The observations all start at an "unassigned"
    /// cluster. At each iteration, the smallest distance in this unassigned class is
    /// evaluated. If both observations are still unassined to a cluster, a new cluster
    /// is created and the two observations are allocated in it. If either observation
    /// is unassigned, it is moved from the unassigned cluster into the cluster where
    /// the other observation is assigned. The algorithm ends when either level (number of
    /// classified observations) is reached, or there are no observations left to be assigned.
    /// The returned vector contains the indices of the observations at each cluster (each
    /// inner vector represents a different cluster).
    fn build_divisive(dist : DMatrix<f64>, level : usize) -> Vec<Vec<usize>> {
        let mut clusters : Vec<Vec<usize>> = Vec::new();
        
        // Build vector of all indices in the upper triangular portion of the matrix,
        // Excluding the diagonal - Those index pairs will be erased as new matches are found.
        let mut unassigned : Vec<Option<(usize, usize)>> = Vec::new();
        for i in 0..dist.nrows() {
            for j in (i+1)..dist.ncols() {
                unassigned.push(Some((i, j)));
            }
        }
        for level in 0..level {
            
            if unassigned.len() == 0 {
                return clusters;
            }
            
            let (mut closest, mut closest_val, mut closest_ix) = ((0, 0), std::f64::INFINITY, 0);
            for (ix, opt_unn) in unassigned.iter().enumerate() {
                if let Some(unn) = opt_unn {
                    if dist[*unn] < closest_val {
                        closest = *unn;
                        closest_val = dist[*unn];
                        closest_ix = ix; 
                    }
                }
            }
            assert!(closest.0 != 0 && closest.1 != 0);
            
            // Remove this pair from the next iterations
            unassigned[closest_ix] = None;
            
            match Self::is_assigned(&clusters[..], closest.0, closest.1) {
                Assignment::Both(_) => {
                    panic!(format!("Pair {:?} should have been removed already", closest));
                },
                Assignment::Row(r_ix) => {
                    clusters[r_ix].push(closest.1);
                }
                Assignment::Col(c_ix) => {
                    clusters[c_ix].push(closest.0);
                } 
                Assignment::Neither => {
                    let mut new_cluster = Vec::new();
                    new_cluster.push(closest.0);
                    new_cluster.push(closest.1);
                    clusters.push(new_cluster);
                }
            }
        }
        clusters
    }
    
    /// Reproduce the dendrogram up to level n_cluster.
    /// If the dendrogram did not achieve this level, return None.
    pub fn level(&self, n_cluster : usize) -> Option<Self> {
        unimplemented!()
    }
    
    /// For the divisive strategy, level means how many individual observations will
    /// be considered for new clusters; for the agglomerative strategy, level means
    /// how many clusters should be formed.
    pub fn build<I,C>(
        data : I,
        strat : Strategy,
        level : usize
    ) -> Self 
    where
        I : Iterator<Item=C>,
        C : AsRef<[f64]>
    {
        let eucl = distance_matrix(data);
        assert!(level >= 2);
        let clusters = match strat {
            Strategy::Divisive => Self::build_divisive(eucl, level),
            Strategy::Agglomerative(_) => unimplemented!()
        };
        Self { clusters }
    }
    
    /*/// Grow the cluster which is closest to the informed data point.
    pub fn grow(&mut self, data : impl AsRef<[f64]>) {
        assert!(self.clusters.len() == 0);
        let mut closest = (0, std::f64::INFINITY);
        let data_v = DVectorSlice::from(data.as_ref());
        for i in 0..self.clusters.len() {
            let diff_norm = (&self.clusters[i].center - data_v).norm();
            if diff_norm < closest.1 {
                closest.0 = i;
                closest.1 = diff_norm;
            }
        }
    }*/
}

#[test]
fn divisive() {

    use crate::distr::MultiNormal;
    
    let s1 = MultiNormal::new_homoscedastic(DVector::from(vec![0.0, 0.0, 0.0]), 0.1);
    let s2 = MultiNormal::new_homoscedastic(DVector::from(vec![0.0, 0.0, 0.0]), 0.1); 
}
