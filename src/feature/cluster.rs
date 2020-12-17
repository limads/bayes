use nalgebra::*;
use std::cmp::{Eq, PartialEq, Ordering};
use std::collections::HashMap;

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

/// Stop the algorithm when either Classified observations have been classified; or
/// the given number of clusters has been formed; or all observations have been classified
pub enum StopCondition {
    Observations(usize),
    Clusters(usize),
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

/// Builds a square euclidian distance matrix from a sequence
/// of observations, each assumed to have the same dimension. Requires that
/// the data is in "tall" form (observations are rows). Passing a wide 
/// matrix saves up one transposition operation.
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

/*/// When iterating over an distance matrix, we can have the row observation
/// assigned to a cluster; the column observation assigned to a cluster; neither, or both.
/// The variants carry the cluster index the observation is assigned to (assuming clusters
/// is a Vec<Vec<usize>> where the usize represents the observation index.
enum Assignment {
    Neither,
    Row(usize),
    Col(usize),
    Both((usize, usize))
}*/

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
    for (i, ref_cluster) in clusters.iter().enumerate() /*.take(clusters.len() / 2)*/ {
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

impl Dendrogram {

    /*/// Returns which observations of the informed pair are already assigned, and at which cluster.
    /// clusters is the vector of observation indices (corresponding to the arrangement at the distance
    /// matrix); obs_row and obs_col is the actual index of the distance matrix.
    fn verify_assignment(clusters : &[Cluster], obs_row : usize, obs_col : usize) -> Assignment {
        let mut row_assigned = None;
        let mut col_assigned = None;
        for (clust_ix, cluster) in clusters.iter().enumerate() {
            if row_assigned.is_none() && cluster.items.iter().find(|o| **o == obs_row ).is_some() {
                row_assigned = Some(clust_ix);
            }
            if col_assigned.is_none() && cluster.items.iter().find(|o| **o == obs_col ).is_some() {
                col_assigned = Some(clust_ix);
            }
        }
        match (row_assigned, col_assigned) {
            (Some(r_ix), Some(c_ix)) => Assignment::Both((r_ix, c_ix)),
            (Some(r_ix), None) => Assignment::Row(r_ix),
            (None, Some(c_ix)) => Assignment::Col(c_ix),
            (None, None) => Assignment::Neither,
        }
    }*/
    
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
    fn build_divisive(dist : DMatrix<f64>, stop : StopCondition) -> Vec<Cluster> {
    
        let ord_dist = Self::build_ordered_distances(&dist);
        println!("Ordered distance: {:?}", ord_dist);
        let mut n_assigned = 0;
        let mut cluster_hash = HashMap::<usize, usize>::new();
        
        // 0th cluster is the background one. Assigned clusters will be associted with
        // indices 1..k
        let mut n_clusters = 0; 
        
        for d in ord_dist {
            match (cluster_hash.get(&d.0), cluster_hash.get(&d.1)) {
                (Some(row_cluster), None) => {
                    n_assigned += 1;
                    cluster_hash.insert(d.1, *row_cluster);
                },
                (None, Some(col_cluster)) => {
                    n_assigned += 1;
                    cluster_hash.insert(d.0, *col_cluster);
                },
                (None, None) => {
                    n_clusters += 1;
                    n_assigned += 2;
                    cluster_hash.insert(d.0, n_clusters);
                    cluster_hash.insert(d.1, n_clusters);
                },
                (Some(_), Some(_)) => { 
                    /*If both are assigned, ignore this distance, since one of the observations
                    can be attributed to a smallest distance, since ord_dist is ordered */
                }
            }
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
        let mut clusters : Vec<Cluster> = Vec::with_capacity(n_clusters);
        for i in 0..(n_clusters) {
            clusters.push(Cluster { items : Vec::new() });
        }
        for (obs, cluster_ix) in cluster_hash.iter() {
            clusters[cluster_ix-1].items.push(*obs);
        }
        clusters
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
        strat : Strategy,
        stop : StopCondition
    ) -> Self 
    where
        I : Iterator<Item=C>,
        C : IntoIterator<Item=&'a f64>
    {
        let wide = build_wide_matrix(data);
        Self::build_from_wide(wide, strat, stop)
    }
    
    /// Run clustering algorithm assuming observations are rows
    pub fn build_from_wide(wide : DMatrix<f64>, strat : Strategy, stop : StopCondition) -> Self {
        assert!(wide.ncols() >= wide.nrows());
        let tall = wide.transpose();
        let eucl = distance_matrix(tall, Some(wide));
        Self::build_from_eucl(eucl, strat, stop)
    }
    
    /// Run clustering algorithm from a prebuilt euclidian distance matrix
    pub fn build_from_eucl(eucl : DMatrix<f64>, strat : Strategy, stop : StopCondition) -> Self {
        println!("Distance matrix shape: {:?}", eucl.shape());
        println!("Distance matrix: {:.3}", eucl);
        let clusters = match strat {
            Strategy::Divisive => Self::build_divisive(eucl, stop),
            Strategy::Agglomerative(_) => panic!("Agglomerative strategy not yet implemented")
        };
        Self { clusters }
    }
    
    /// Run clustering algorithm assuming observations are columns
    pub fn build_from_tall(tall : DMatrix<f64>, strat : Strategy, stop : StopCondition) -> Self {
        assert!(tall.ncols() <= tall.nrows());
        let eucl = distance_matrix(tall, None);
        Self::build_from_eucl(eucl, strat, stop)
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
    use bayes::feature::cluster::*;
    use bayes::prob::*;

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
        Strategy::Divisive, 
        StopCondition::All
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


