// TODO remove this file. It has been moved to away::dendro::mod.

use nalgebra::*;
use std::cmp::{Eq, PartialEq, Ordering};
use std::collections::HashMap;

// TODO perhaps make it Cluster<T> where T stores the indices and a a &[T] to a parent
// clustering algorithm strategy that also stores the distance matrix.
#[derive(Debug, Clone)]
pub struct Cluster {
    // Observations are assumed to have a natural order corresponding to their entry in the
    // distance matrix. Each cluster (inner vector) stores the index of the observation.
    items : Vec<usize>
}

impl Cluster {

    pub fn items(&self) -> &[usize] {
        &self.items[..]
    }
    
}

impl AsRef<[usize]> for Cluster {
    fn as_ref(&self) -> &[usize] {
        self.items()
    }
}

/// For an agglomerative clustering algorithm, this enum identifies
/// which distance between pairs of elements in each cluster will be used to
/// define the cluster fusion: In the single linkage strategy (Nearest neighbor), the distance
/// between the closest elements (each from one of the cluster) will be used;
/// In the complete linkage strategy (Farthest neighbor), the distance between the farthest elements (each from
/// one of the clusters) will be used.
#[derive(Debug, Clone, Copy)]
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
/// clusters of the same size (although not necessarily the "best" ones). If the mixed
/// strategy is chosen, the algorithm starts with a divisive strategy, then applies an agglomerative
/// strategy to the output of the first step. In the threshold strategy, we order the full distance
/// matrix. Assign to current cluster as long as the current distance is smaller than some threhsold;
/// or innaugurate a new cluster when this distance is greater than the threshold.
pub enum Strategy {
    Divisive(StopCondition),
    Threshold(f64),
    Agglomerative(Neighbor, StopCondition),
    Mixed(StopCondition, (Neighbor, StopCondition))
}

/// Stop the algorithm when either Classified observations have been classified; or
/// the given number of clusters has been formed; or all observations have been classified
pub enum StopCondition {

    // In a divise strategy, how many observations are NOT on the "background" cluster.
    // In an agglomerative strategy, when the greatest cluster size is achieved.
    Observations(usize),
    
    // Stop when we have exactly the given number of clusters
    Clusters(usize),
    
    // Stop when we have between n and m clusters, returning the size with the highest separability
    // ClusterRange(usize, usize),
    
    // In a divise strategy, stop when there are no observations left on the background cluster.
    // In an agglomerative strategy, returns a single cluster for all observations.
    All
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
    clusters : Vec<Cluster>
}

// Build the "wide" data matrix (observations are columns) from an observation iterator.
fn build_wide_matrix<'a, I, C>(obs : I) -> DMatrix<f64> 
where
    I : Iterator<Item=C>,
    C : IntoIterator<Item=&'a f64>
{
    let vecs : Vec<_> = obs
        .map(|item| DVector::from_vec(item.into_iter().cloned().collect::<Vec<f64>>()) )
        .collect();
    DMatrix::from_columns(&vecs[..])
}

/// Builds a generic distance matrix from a arbitrary metric defined for a given structure.
pub fn generic_distance_matrix<T, F>(obs : &[T], metric : F) -> DMatrix<f64>
where
    F : Fn(&T, &T)->f64
{
    let n = obs.len();
    let mut dist = DMatrix::zeros(n, n);

    // Store measure entry results in upper-triangular portion
    for i in 0..n {
        for j in i..n {
            dist[(i, j)] = metric(&obs[i], &obs[j]);
        }
    }

    // Copy entries to lower-triangular portion
    for j in 0..n {
        for i in 0..j {
            dist[(j, i)] = dist[(i, j)]
        }
    }
    dist
}

/// Builds a square euclidian distance matrix from a sequence
/// of observations, each assumed to have the same dimension. Requires that
/// the data is in "tall" form (observations are rows). Passing a wide 
/// matrix saves up one transposition operation. The crate strsim-rs
/// can be used to build custom text metrics. The user can write any
/// Fn(&T, &T)->f64 to serve as a custom object metric.
pub fn distance_matrix(tall : DMatrix<f64>, opt_wide : Option<DMatrix<f64>>) -> DMatrix<f64> {
    let wide = opt_wide.unwrap_or(tall.transpose());
    
    // Build the Gram matrix (Matrix of dot products)
    // Reference: https://en.wikipedia.org/wiki/Euclidean_distance_matrix#Relation_to_Gram_matrix
    let gram = tall * wide;
    
    assert!(gram.nrows() == gram.ncols());
    
    // Calculate the Euclidian distance matrix from the Gram matrix
    let mut eucl = DMatrix::zeros(gram.nrows(), gram.ncols());
    for i in 0..gram.nrows() {
        for j in 0..gram.ncols() {
            eucl[(i, j)] = gram[(i, i)] - 2. * gram[(i, j)] + gram[(j, j)];
        }
    }
    
    // All diagonals must be zero (Eucl. dist. matrix is hollow)
    for i in 0..eucl.nrows() {
        assert!(eucl[(i,i)] < 1E-8);
    }
    
    // All distances must be non-negative
    for i in 0..eucl.nrows() {
        for j in (i + 1)..eucl.ncols() {
            assert!(eucl[(i, j)] >= 0.0);
        }
    }
    
    // Matrix must be symmetric
    for i in 0..eucl.nrows() {
        for j in 0..eucl.ncols() {
            assert!(eucl[(i, j)] - eucl[(j, i)] < 1E-8);
        }
    }
    
    eucl
}

/*
// closest_ix index the unassigned vector; closest is the actual distance matrix index
    let (mut closest, mut closest_val, mut closest_ix) = ((0, 0), std::f64::INFINITY, 0);
    for (cmp_ix, opt_unn) in unassigned.iter().enumerate() {
    
        // Ignore entries which are None in the unassigned vector - Iterate only over
        // assigned ones.
        if let Some(unn) = opt_unn {
            if dist[*unn] < closest_val {
                closest = *unn;
                closest_val = dist[*unn];
                closest_ix = cmp_ix; 
            }
        }
    }
    
    // This block is for debugging only
    sorted_vals.push(closest_val);
    // assert!(sorted_vals.is_sorted());
    let mut sorted_clone = sorted_vals.clone();
    sorted_clone.sort_by(|c1, c2| c1.partial_cmp(c2).unwrap_or(Ordering::Equal) );
    assert!(sorted_vals.iter().zip(sorted_clone.iter()).all(|(c1, c2)| (*c1 - *c2) < 1E-8 ));
    
    // Remove this unassigned pair from the next iteration
    unassigned[closest_ix] = None;
*/

/// Finds the "best" neighbor, where best is defined as the closest pair of neighbors in
/// the nearest strategy; or the farthest pair of neighbors in the farthest strategy.
fn best_neighbor_dist(
    dist : &DMatrix<f64>, 
    ref_cluster : &Cluster, 
    cand_cluster : &Cluster,
    neighbor : &Neighbor
) -> f64 {
    let mut neighbor_dist = match neighbor {
        Neighbor::Farthest => 0.0,
        Neighbor::Nearest => std::f64::INFINITY
    };
    for ref_ix in ref_cluster.items.iter() {
        for cand_ix in cand_cluster.items.iter() {
            let d = dist[(*ref_ix, *cand_ix)];
            match neighbor {
                Neighbor::Farthest => if d > neighbor_dist {
                    neighbor_dist = d;
                },
                Neighbor::Nearest => if d < neighbor_dist {
                    neighbor_dist = d;
                }
            }
        }
    }
    neighbor_dist
}

/// Trim the smallest cluster to leave an even number of clusters. Do nothing to the 
/// cluster vector otherwise.
pub fn trim_odd_cluster(clusters : &mut Vec<Cluster>) {
    if clusters.len() % 2 != 0 {
        let mut smallest = std::usize::MAX;
        let mut smallest_ix = 0;
        for (i, clust) in clusters.iter().enumerate() {
            let n = clust.items().len();
            if n < smallest {
                smallest = n;
                smallest_ix = i;
            }
        }
        clusters.remove(smallest_ix);
    }
}

/// Given a cluster vector of size N (assumed even), returns a cluster vector of size n/2 by aggregating
/// all cluster pair-wise, based either on the nearest neighbor or farthest neighbor rule. Panics if an odd number
/// of clusters is passed.
pub fn agglomerate(dist : &DMatrix<f64>, clusters : &[Cluster], neighbor : Neighbor) -> Vec<Cluster> {
    assert!(clusters.len() % 2  == 0);
    
    let mut ord_dist = Vec::new();
    for (i, ref_cluster) in clusters.iter().enumerate() {
        for (j, cand_cluster) in clusters.iter().enumerate().skip(i+1) {
            let cluster_dist = best_neighbor_dist(&dist, &ref_cluster, &cand_cluster, &neighbor);
            ord_dist.push((i, j, cluster_dist));
        }        
    }
    
    ord_dist.sort_by(|da, db| da.2.partial_cmp(&db.2).unwrap_or(Ordering::Equal) );
    
    let agg_len = clusters.len() / 2;
    let mut agg_clusters = Vec::with_capacity(agg_len);
    for (ix_a, ix_b, _) in &ord_dist[0..agg_len] {
        let mut items = clusters[*ix_a].items.clone();
        items.extend(clusters[*ix_b].items.iter());
        agg_clusters.push(Cluster{ items : items.to_vec() } );
    }
    agg_clusters
}

/// Background cluster is cluster zero.
fn build_cluster_vector(cluster_hash : HashMap<usize, usize>, n_clusters : usize) -> Vec<Cluster> {
    let mut clusters : Vec<Cluster> = Vec::with_capacity(n_clusters);
    clusters.extend((0..n_clusters).map(|_| Cluster { items : Vec::new() }));
    for (obs, cluster_ix) in cluster_hash.iter() {
        clusters[cluster_ix-1].items.push(*obs);
    }
    clusters
}

/// Walk one iteration of the divisive clustering strategy.
fn cluster_next_divisive(
    cluster_hash : &mut HashMap<usize, usize>,
    row_ix : usize,
    col_ix : usize,
    n_clusters : &mut usize,
    n_assigned : &mut usize
) {
    match (cluster_hash.get(&row_ix), cluster_hash.get(&col_ix)) {
        (Some(row_cluster), None) => {
            *n_assigned += 1;
            cluster_hash.insert(col_ix, *row_cluster);
        },
        (None, Some(col_cluster)) => {
            *n_assigned += 1;
            cluster_hash.insert(row_ix, *col_cluster);
        },
        (None, None) => {
            *n_clusters += 1;
            *n_assigned += 2;
            cluster_hash.insert(row_ix, *n_clusters);
            cluster_hash.insert(col_ix, *n_clusters);
        },
        (Some(_), Some(_)) => {
            /*If both are assigned, ignore this distance, since one of the observations
            can be attributed to a smallest distance, since ord_dist is ordered */
        }
    }
}

impl Dendrogram {

    // Build sparse representation of upper triangular portion of the matrix as a Vec of (row, col, value),
    // ordered by smallest to largest observation distance.
    fn build_ordered_distances(dist : &DMatrix<f64>) -> Vec<(usize, usize, f64)> {
        let mut ord_dist : Vec<(usize, usize, f64)> = Vec::with_capacity(dist.nrows().pow(2) / 2);
        for i in 0..dist.nrows() {
            for j in (i+1)..dist.ncols() {
                ord_dist.push((i, j, dist[(i, j)]));
            }
        }
        ord_dist.sort_by(|da, db| da.2.partial_cmp(&db.2).unwrap_or(Ordering::Equal) );
        ord_dist
    }
    
    fn build_threshold(dist : &DMatrix<f64>, threshold : f64) -> Vec<Cluster> {
        let ord_dist = Self::build_ordered_distances(&dist);

        // Observation indices are keys; cluster indices are columns
        let mut cluster_hash = HashMap::<usize, usize>::new();
        let (mut n_clusters, mut n_assigned) = (0, 0);

        for (row_ix, col_ix, dist) in ord_dist {
            if dist < threshold {
                // If distance is smaller than threshold, aggregate to the closest cluster, or
                // innaugurate a new one only if neither elements are assigned. This will create
                // new clusters sparingly, only when neither elements are aggregated.
                cluster_next_divisive(&mut cluster_hash, row_ix, col_ix, &mut n_clusters, &mut n_assigned);
            } else {
                // If distance is largest than threshold, innaugurate a new cluster with
                // the unassigned element (or both unassigned elements if neither are assigned).
                // This branch will create several single-element clusters, because n_clusters is
                // incremented irrespective of whether elements are assigned or not.
                // Since element distance is below threhsold, insert them separately.
                if cluster_hash.get(&row_ix).is_none() {
                    n_clusters += 1;
                    cluster_hash.insert(row_ix, n_clusters);
                }

                if cluster_hash.get(&col_ix).is_none() {
                    n_clusters += 1;
                    cluster_hash.insert(col_ix, n_clusters);
                }
            }
        }
        build_cluster_vector(cluster_hash, n_clusters)
    }

    fn build_agglomerative(
        opt_clusters : Option<Vec<Cluster>>,
        dist : &DMatrix<f64>, 
        stop : StopCondition, 
        neighbor : Neighbor
    ) -> Vec<Cluster> {
        let mut clusters : Vec<Cluster> = opt_clusters
            .unwrap_or_else(|| (0..dist.nrows()).map(|ix| Cluster{ items : vec![ix] } ).collect());
        loop {
            match stop {
                StopCondition::Observations(n_obs) => {
                    panic!("Observaitons conditions not applicable to agg strategy")
                },
                StopCondition::Clusters(n_clust) => {
                    if clusters.len() >= n_clust {
                        return clusters;
                    }
                },
                StopCondition:: All => {
                    if clusters.len() == 1 {
                        return clusters;
                    }        
                }
            }
            clusters = agglomerate(&dist, &clusters[..], neighbor);
        }
        clusters
    }
    
    /// Builds a dendrogram using a divisive strategy, until a number of observations
    /// given by level are classified. The observations all start at an "unassigned"
    /// cluster. At each iteration, the smallest distance in this unassigned class is
    /// evaluated. If both observations corresponding to this distance are still 
    /// unassined to a cluster, a new cluster is created and the two observations are allocated
    /// in it. If either observation is unassigned, it is moved from the unassigned cluster 
    /// into the cluster where the other observation is assigned. The algorithm ends when either level 
    /// (number of classified observations) is reached, or there are no observations left to be assigned.
    /// The returned vector contains the indices of the observations at each cluster (each
    /// inner vector represents a different cluster).
    fn build_divisive(dist : &DMatrix<f64>, stop : StopCondition) -> Vec<Cluster> {
        let ord_dist = Self::build_ordered_distances(&dist);
        
        // Observation indices are keys; cluster indices (1..k) are columns
        let mut cluster_hash = HashMap::<usize, usize>::new();
        let (mut n_clusters, mut n_assigned) = (0, 0);
        
        for (row_ix, col_ix, _) in ord_dist {
            cluster_next_divisive(&mut cluster_hash, row_ix, col_ix, &mut n_clusters, &mut n_assigned);
            match &stop {
                StopCondition::Clusters(n_clust) => if n_clusters == *n_clust {
                    break;
                },
                StopCondition::Observations(n_obs) => if n_assigned == *n_obs {
                    break;
                },
                StopCondition::All => { }
            }
        }
        
        assert!(n_assigned == dist.nrows());
        
        // Ignore potentially unclassified background observations
        build_cluster_vector(cluster_hash, n_clusters)
    }
    
    /*/// Reproduce the dendrogram up to level n_cluster.
    /// If the dendrogram did not achieve this level, return None.
    pub fn level(&self, n_cluster : usize) -> Option<Self> {
        unimplemented!()
    }*/
    
    pub fn clusters(&self) -> &[Cluster] {
        &self.clusters[..]
    }
    
    /// Run clustering algorithm from iterator over variables nested within iterator over observations.
    pub fn build<'a, I,C>(
        data : I,
        strat : Strategy
    ) -> Self 
    where
        I : Iterator<Item=C>,
        C : IntoIterator<Item=&'a f64>
    {
        let wide = build_wide_matrix(data);
        Self::build_from_wide(wide, strat)
    }
    
    /// Run clustering algorithm assuming observations are rows
    pub fn build_from_wide(wide : DMatrix<f64>, strat : Strategy) -> Self {
        assert!(wide.ncols() >= wide.nrows());
        let tall = wide.transpose();
        let eucl = distance_matrix(tall, Some(wide));
        Self::build_from_eucl(eucl, strat)
    }
    
    pub fn build_generic<T, F>(obs : &[T], metric : F, strat : Strategy) -> Self
    where
        F : Fn(&T, &T)->f64
    {
        let dist = generic_distance_matrix(obs, metric);
        Self::build_from_eucl(dist, strat)
    }

    /// Run clustering algorithm from a prebuilt euclidian distance matrix
    pub fn build_from_eucl(eucl : DMatrix<f64>, strat : Strategy) -> Self {
        println!("Distance matrix shape: {:?}", eucl.shape());
        println!("Distance matrix: {:.3}", eucl);
        let clusters = match strat {
            Strategy::Divisive(stop) => {
                Self::build_divisive(&eucl, stop)
            },
            Strategy::Agglomerative(neighbor, stop) => {
                Self::build_agglomerative(
                    None,
                    &eucl, 
                    stop, 
                    neighbor
                )
            },
            Strategy::Mixed(div_stop, (neighbor, agg_stop)) => {
                let div_clusters = Self::build_divisive(&eucl, div_stop);
                Self::build_agglomerative(
                    Some(div_clusters),
                    &eucl, 
                    agg_stop, 
                    neighbor
                )
            },
            Strategy::Threshold(thresh) => {
                Self::build_threshold(&eucl, thresh)
            }
        };
        Self { clusters }
    }
    
    /// Run clustering algorithm assuming observations are columns
    pub fn build_from_tall(tall : DMatrix<f64>, strat : Strategy) -> Self {
        assert!(tall.ncols() <= tall.nrows());
        let eucl = distance_matrix(tall, None);
        Self::build_from_eucl(eucl, strat)
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
fn cluster() {

    use nalgebra::*;
    use crate::feature::cluster::*;
    use crate::prob::*;

    let mut clust_gen = Vec::new();
    let mut data = DMatrix::zeros(40, 2);
    for i in 0..4 {
        let mu = DVector::from_column_slice(&[i as f64 * 10.0, i as f64 * 20.0]);
        let sigma = DMatrix::from_column_slice(2, 2, &[0.001, 0.0, 0.0, 0.001]);
        clust_gen.push(MultiNormal::new(10, mu, sigma).unwrap());
        clust_gen[i].sample_into(data.slice_mut((10*i, 0), (10, 2)));
    }
    println!("Observations: {}", data);
    let dist = distance_matrix(data.clone(), None);
    let dendr = Dendrogram::build_from_tall(
        data,
        Strategy::Divisive(StopCondition::All), 
    );
    
    let mut clusters : Vec<Cluster> = dendr.clusters().to_vec();
    trim_odd_cluster(&mut clusters);
    
    println!("Clusters: {:?}", clusters);
    println!("Clusters trimmed length: {}", clusters.len());
    let agg = agglomerate(&dist, &clusters, Neighbor::Farthest);
    println!("Agglomerated clusters: {:?}", agg);
    println!("Agglomerated length: {}", agg.len());
    for cluster in agg {
        let min = cluster.items().iter().min().unwrap();
        let max = cluster.items().iter().max().unwrap();
        assert!( (*max as i32) - (*min as i32) < 10);
    }
}

/// Calculates the centroid of the informed cluster, from the full data matrix.
pub fn centroid_mut(m : &DMatrix<f64>, cluster : &Cluster, centroid : &mut DVector<f64>) {
    let n = cluster.items.len() as f64;
    centroid.iter_mut().for_each(|c| *c = 0.0 );
    for (c, v) in centroid.iter_mut().zip(cluster.items().iter()) {
        *c += *v as f64 / n;
    }
}

/*
/// A metric is a scalar value that captures the distance or dissimilarity between
/// a pair of n-dimensional observations. If those observations are arranged into
/// a Sample implementor, within(.) returns the square matrix of pair-wise metrics
/// of all observations within the sample. Metrics can also be calculated as the
/// pair-wise comparison of all elements of a pair of sample implementors via the between(.)
/// method.
///
/// Metrics are an important general-purpose dimensionality reduction algorithm, which
/// always reduce the dimension from n to 1. The dissimilarity between an observation and
/// a series of prototypes can be used for classification; the dissimilarity between the cartesian
/// product of a pair of sets of observations can be used for matching and clustering. Clustering
/// and Matching algorihtms are generic over the metric they use. Clustering algorithms take only
/// the metric and output the clusters or interst; Matching algoritms take the metric and an optimizer
/// (function of the observation match) and output the state of the optimizer that best satisfies
/// all the pair-wise dissimilarities.
pub trait Metric
where Self : Sized
{

    fn dim(&self) -> usize;

    fn within<S>(a : S) -> Self
    where S : Into<DMatrix<f64>> + Clone
    {
        Self::between(a.clone(), a)
    }

    // TODO use set here.
    fn between<S>(a : S, b : S) -> Self
    where S : Into<DMatrix<f64>>;

    /// Returns the full distance matrix. For within-sample comparisons,
    /// (Self::within), the diagonal values will always be zero, since the
    /// elements are being compared to themselves at the i==j entries.
    fn full(&self) -> DMatrix<f64>;

    /// Return a matrix with the n distances closest to the informed point.
    /// The row index correspond to the observation index at left set;
    /// the column index correspond to the observation index at the
    /// right set. The implementor should guarantee that if a pair-wise
    /// comparison is being made (Self::within), the comparison of an
    /// element with itself is not returned (the sparse matrix will never
    /// have an element in the diagonal).
    fn closest_to(&self, pt : &[f64], n : usize) -> CsMatrix<f64>;

    /// Return a sparse matrix containing the n-smallest distances.
    fn smallest(&self, n : usize) -> CsMatrix<f64> {
        let mut pt : Vec<f64> = Vec::new();
        pt.extend((0..self.dim()).map(|_| 0.0 ));
        self.closest_to(&pt[..], n)
    }

    // TODO add provided method for histogram.
}

/// Represents an upper triangular matrix of euclidian distances
pub struct Euclidian {
    dim : usize,
    dst : DMatrix<f64>
}

/// Represents an upper triangular matrix of Manhattan distances
pub struct Manhattan {

}

impl Metric for Manhattan {

    fn between<S>(a : S, b : S) -> Self
        where S : Into<DMatrix<f64>>
    {
        unimplemented!()
    }

    fn full(&self) -> DMatrix<f64> {
        unimplemented!()
    }

    fn dim(&self) -> usize {
        unimplemented!()
    }

    fn closest_to(&self, pt : &[f64], n : usize) -> CsMatrix<f64> {
        unimplemented!()
    }

}

impl Metric for Euclidian {

    fn dim(&self) -> usize {
        self.dim
    }

    fn between<S>(a : S, b : S) -> Self
        where S : Into<DMatrix<f64>>
    {
        let a : DMatrix<f64> = a.into();
        let b : DMatrix<f64> = b.into();
        assert!(a.ncols() == b.ncols());
        let mut dst = DMatrix::zeros(a.nrows(), b.ncols());
        for (i, row_a) in a.row_iter().enumerate() {
            for (j, row_b) in b.row_iter().enumerate() {
                dst[(i, j)] = row_a.iter()
                    .zip(row_b.iter())
                    .fold(0.0, |sum, p| sum + (p.0.powf(2.) - p.1.powf(2.)).abs() as f64)
                    .sqrt();
            }
        }
        Self{ dst, dim : a.ncols() }
    }

    fn full(&self) -> DMatrix<f64> {
        self.dst.clone()
    }

    fn closest_to(&self, pt : &[f64], n : usize) -> CsMatrix<f64> {
        unimplemented!()
    }

}
*/
