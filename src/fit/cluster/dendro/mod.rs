use crate::fit::cluster::{Distance, Metric};
use itertools::Itertools;
use std::cmp::Ordering;
use nalgebra::DMatrix;
use crate::fit::cluster;

pub struct Dendrogram<'a, T> {
    pts : &'a [T],
    clusters : Vec<Vec<usize>>
}

// Iterate over clusters (element is &[usize]).
// impl Iterator for Dendrogram

pub enum Linkage {
    Partial,
    Full
}

#[derive(Debug, Clone, Copy)]
struct Link {
    src_clust : usize,
    dst_clust : usize,
    clust_dist : f64
}

fn different_clusters(link_a : &Link, link_b : &Link) -> bool {
    link_a.src_clust != link_b.src_clust &&
        link_a.src_clust != link_b.dst_clust &&
        link_a.dst_clust != link_b.src_clust &&
        link_a.dst_clust != link_b.dst_clust
}

fn hierarchical_clustering<'a, T, M>(
    pts : &'a [T],
    linkage : Linkage,
    n_clust : usize
) -> Dendrogram<'a, T>
where
    T :  Distance<M>
{

    assert!(pts.len() % 2 == 0);
    let mut links : Vec<Link> = Vec::new();

    // Each element holds the indices of pts allocated to the cluster.
    let mut clusters : Vec<Vec<usize>>= (0..pts.len()).map(|ix| vec![ix] ).collect();

    let dist = cluster::distance::generic_distance_matrix(pts, |a, b| { a.distance(b) }, false);

    for level in 0..n_clust {
        update_links(&mut links, &clusters[..], &dist, &linkage);
        aggregate(&mut clusters, &mut links);
    }

    Dendrogram {
        pts,
        clusters
    }
}

fn update_links(links : &mut Vec<Link>, clusts : &[Vec<usize>], dist : &DMatrix<f64>, linkage : &Linkage) {
    links.clear();
    for (ix_a, clust_a) in clusts.iter().enumerate() {
        for (ix_b, clust_b) in clusts.iter().enumerate() {
            if ix_a == ix_b {
                continue;
            }
            let dist_iter = clust_a.iter().cartesian_product(clust_b.iter())
                .map(|(ix_a, ix_b)| dist[(*ix_a, *ix_b)] );
            let clust_dist = match linkage {
                Linkage::Full => {
                    dist_iter.min_by(|d_a, d_b| d_a.partial_cmp(&d_b).unwrap_or(Ordering::Equal) ).unwrap()
                },
                Linkage::Partial => {
                    dist_iter.max_by(|d_a, d_b| d_a.partial_cmp(&d_b).unwrap_or(Ordering::Equal) ).unwrap()
                }
            };
            links.push(Link { src_clust : ix_a, dst_clust : ix_b, clust_dist });
        }
    }
}

fn aggregate(clusts : &mut Vec<Vec<usize>>, links : &mut Vec<Link>) {
    let n_clust = clusts.len();
    let mut ix = 0;
    let mut n_aggregated = 0;
    let mut invalidated_clusters = Vec::new();
    while n_aggregated < n_clust / 2 {
        let opt_best_link = links.iter()
            .filter(|link| (link.src_clust == ix || link.dst_clust == ix) )
            .min_by(|link_a, link_b| link_a.clust_dist.partial_cmp(&link_b.clust_dist).unwrap_or(Ordering::Equal) );
        if let Some(best_link) = opt_best_link.cloned() {
            let old_cluster = clusts[best_link.src_clust].clone();
            clusts[best_link.dst_clust].extend(old_cluster);
            invalidated_clusters.push(best_link.src_clust);
            links.retain(|link| different_clusters(&link, &best_link) );
            n_aggregated += 1;
        }
        ix += 1;
    }
    for inv in invalidated_clusters {
        clusts.swap_remove(inv);
    }
}


