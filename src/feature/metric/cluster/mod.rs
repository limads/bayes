use nalgebra::*;
use super::*;

pub struct Cluster {

    /// Rows of sample pertaining to this cluster.
    pub items : Vec<usize>,

    // mean and cov can be recovered from the
    // SVD of elements within each cluster if only
    // the distances are preserved.
    pub mean : Option<DVector<f64>>,
    pub cov : Option<DMatrix<f64>>
}

/// K-means cluster objects by minimizing their metric
/// with respect to a prototype or centroid object, built
/// from the average of the observations.
pub struct KMeans {

    /// Indices of the sample columns that were used for clustering.
    cols : usize

}

/// The hierarchical clustering algorithm generate clusters
/// by minimizing the pair-wise metric of the closest or
/// farthest element from a cluster and a test observation.
pub struct HClust {

    /// Distance/Proximity of observation i (row) to j (column).
    metrics : DMatrix<f64>,

    /// Indices of the sample columns that were used for clustering.
    cols : usize
}

impl HClust {

    /// After setting the level, iter_clusters() will
    /// iterate at the corresponding level in the dendrogram.
    pub fn set_level(&mut self, lvl : usize) {

    }
}

/// Trait implemented by clustering algorithms.
pub trait Clustering<'a, M : Metric> {

    fn cluster<D>(&'a mut self, d : D) -> Result<&'a [Cluster], String>
        where D : Into<DMatrix<f64>>;

    fn view_clusters(&'a self) -> &'a [Cluster];

}


