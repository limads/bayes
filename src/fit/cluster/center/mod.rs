use super::*;
use nalgebra::DVector;
use std::borrow::Borrow;
use crate::fit::Estimator;
use std::fmt::Display;
use rand::prelude::*;
use std::fmt;
use std::collections::HashMap;
use nalgebra::DVectorSlice;

pub struct KMeansSettings {
    pub n_cluster : usize,
    pub max_iter : usize,
    pub allocations : Option<Vec<usize>>
}

#[derive(Debug)]
pub struct KMeansError;

/// K-means is a special case of the EM algorithm for GMMs where
/// the probabilities of class allocation are assumed constant and equal;
/// and the covariances within clusters are assumed constant and equal as well.
#[derive(Debug)]
pub struct KMeans {
    allocations : Vec<usize>,
    means : Vec<DVector<f64>>,
    n_iter : usize
}

impl KMeans {

    /// Returns mean of each cluster, with order matching the indices returned by Self::allocations
    pub fn means(&self) -> impl Iterator<Item=&[f64]> {
        self.means.iter().map(|m| m.as_slice() )
    }

    /// Return cluster of each observation, in the order they were supplied
    pub fn allocations(&self) -> &[usize] {
        &self.allocations[..]
    }

    pub fn iterations(&self) -> usize {
        self.n_iter
    }

    pub fn count_allocations(&self, cluster_ix : usize) -> usize {
        self.allocations().iter().filter(|alloc| **alloc == cluster_ix ).count()
    }

}

/// In case the observations had size one, returns the minimum and
/// maximum observations assigned to the given cluster. This function
/// is not a part of KMeans, since it would require storing the samples
/// inside the structure, which might impact performance.
pub fn extremes(
    km : &KMeans,
    sample : impl Iterator<Item=impl Borrow<[f64]>> + Clone,
    cluster_ix : usize
) -> Option<(f64, f64)> {
    if km.means[0].len() > 1 {
        return None;
    }
    let (mut min, mut max) = (f64::INFINITY, 0.0);
    if sample.clone().count() != km.allocations().len() {
        return None;
    }
    for (alloc, sample) in km.allocations().iter().zip(sample) {
        if *alloc == cluster_ix {
            let s = sample.borrow();
            if s[0] > max {
                max = s[0];
            }
            if s[0] < min {
                min = s[0];
            }
        }
    }
    Some((min, max))
}

impl fmt::Display for KMeans {

    fn fmt(&self, f : &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let mut dst = String::new();
        for c in 0.. self.means.len() {
            let obs_ix = self.allocations.iter()
                .enumerate()
                .filter(|(_, alloc)| **alloc == c )
                .map(|(ix, _)| ix )
                .collect::<Vec<usize>>();
            dst += &format!("Cluster {}: Mean = {}; Observations = {:?}\n", c, self.means[c], obs_ix);
        }
        write!(f, "{}", dst)
    }
}

/// Other estimation algorithm for categorical classification.
pub struct KNearest {

}

pub struct ClusterInfo {
    sum : DVector<f64>,
    count : usize,
}

fn euclidian(a : &[f64], b : &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powf(2.) ).sum::<f64>().sqrt()
}

fn update_means(
    means : &mut [DVector<f64>],
    clusters : &mut HashMap<usize, ClusterInfo>,
    sample : impl Iterator<Item=impl Borrow<[f64]>> + Clone,
    allocations : &[usize],
    settings : &KMeansSettings
) {
    assert!(allocations.len() == sample.clone().count());

    let obs_dim = sample.clone().next().unwrap().borrow().len();

    // Clear previous iteration info
    for c in 0..settings.n_cluster {
        means[c] = DVector::zeros(obs_dim);
        let mut cluster = clusters.get_mut(&c).unwrap();
        cluster.sum = DVector::zeros(obs_dim);
        cluster.count = 0;
    }

    // Accumulate each observation into sum and count
    for (ix, obs) in sample.enumerate() {
        let mut cluster = clusters.get_mut(&allocations[ix]).unwrap();
        cluster.sum += DVectorSlice::from(obs.borrow());
        cluster.count += 1;
    }

    // Calculate averages from sum and accumulated count
    for c in 0..settings.n_cluster {
        means[c] = clusters[&c].sum.clone();
        means[c].scale_mut(1. / (clusters[&c].count as f64));
    }

}

fn closest<'a>(candidates : impl Iterator<Item=&'a (impl AsRef<[f64]> + 'a + ?Sized)>, el : &'a [f64]) -> usize {
    let mut min_dist = f64::INFINITY;
    let mut min_ix = 0;
    for (cand_ix, cand) in candidates.enumerate() {
        let dist_to_cand = euclidian(el, cand.as_ref());
        if dist_to_cand < min_dist {
            min_ix = cand_ix;
            min_dist = dist_to_cand;
        }
    }
    min_ix
}

fn furthest<'a>(candidates : impl Iterator<Item=&'a (impl AsRef<[f64]> + 'a + ?Sized)>, el : &'a [f64]) -> usize {
    let mut max_dist = 0.0;
    let mut max_ix = 0;
    for (cand_ix, cand) in candidates.enumerate() {
        let dist_to_cand = euclidian(el, cand.as_ref());
        if dist_to_cand > max_dist {
            max_ix = cand_ix;
            max_dist = dist_to_cand;
        }
    }
    max_ix
}

/// Update allocation vector, returning how many observations were re-allocated.
fn update_allocations(
    allocations : &mut [usize],
    means : &[DVector<f64>],
    sample : impl Iterator<Item=impl Borrow<[f64]>> + Clone
) -> usize {
    assert!(allocations.len() == sample.clone().count());
    let mut n_reallocated = 0;
    for (mut alloc, obs) in allocations.iter_mut().zip(sample) {
        let best_alloc = closest(means.iter().map(|v| v.as_slice() ), obs.borrow());
        if *alloc != best_alloc {
            *alloc = best_alloc;
            n_reallocated += 1;
        }
    }
    n_reallocated
}

fn take_random_while_not_in(prev : &mut Vec<usize>, n : usize) -> usize {

    // If this condition is violated, our program could run forever if all elements
    // in prev are distinct in the set {0..n}
    assert!(prev.len() < n);

    loop {
        let rand_obs = (0..n).choose(&mut rand::thread_rng()).unwrap();
        if !prev.iter().any(|p| *p == rand_obs) {
            prev.push(rand_obs);
            return rand_obs;
        }
    }
}

fn random_allocations(settings : &KMeansSettings, n : usize) -> Vec<usize> {

    // Start by doing a random assignment
    let mut allocations : Vec<usize> = (0..n).map(|ix|
        (0..settings.n_cluster).choose(&mut rand::thread_rng()).unwrap()
    ).collect();

    // Make sure at least one observation is assigned to each cluster by hard-setting
    // the cluster index of n_cluster random observations over the previously-generated values.
    let mut hard_set = Vec::new();
    for c in 0..settings.n_cluster {
        let rand_obs = take_random_while_not_in(&mut hard_set, n);
        allocations[rand_obs] = c;
    }

    allocations
}

/// Seeding (choosing a representative observation for each cluster
/// before starting the KMeans algorithm) helps with the identifiability
/// of the model, biasing the algorithm to one of the equally likely
/// cluster allocations. If cluster happens to the tightly packed together,
/// this stage already imposes some structure that KMeans can explore,
/// since all allocations are defined based on proximity to each seed cluster.
fn seed_allocations(
    settings : &KMeansSettings,
    sample : impl Iterator<Item=impl Borrow<[f64]>> + Clone,
    n : usize
) -> Vec<usize> {

    // At the ith index, holds a representative observation off the ith cluster.
    let mut seeds : Vec<(usize, DVector<f64>)> = Vec::with_capacity(settings.n_cluster);

    // Abosolutely random seeds. Might work when we have many clusters, where the
    // probability of taking two seeds at the same cluster is small.
    /*let mut seed_ixs = Vec::with_capacity(settings.n_cluster);
    for c in 0..settings.n_cluster {
        let rand_obs = take_random_while_not_in(&mut seed_ixs, n);
        let mut sc = sample.clone().nth(rand_obs).unwrap();
        let sb = sc.borrow();
        seeds.push((rand_obs, DVector::from_iterator(sb.len(), sb.iter().cloned() )));
    }*/

    /*// Take first seed as random. Then Take next seed to be the furthest from average of last seeds.
    // This increase the chances of taking seeds that are far apart.
    for c in 0..settings.n_cluster {
        let rand_obs = if c == 0 {
            (0..n).choose(&mut rand::thread_rng()).unwrap()
        } else {
            let mut avg_seeds = DVector::zeros(settings.n_cluster);
            for (_, obs) in seeds.iter() {
                avg_seeds += DVectorSlice::from(obs.borrow());
            }
            avg_seeds.scale_mut(1. / (settings.n_cluster as f64));

            let mut furthest_obs = (0, 0.0);
            for (ix, obs) in sample.clone() {
                let obs_is_seed = seeds.iter().any(|(s_ix, _)| s_ix == ix );
                if !obs_is_seed {
                    let dist = euclidian(obs.borrow(), avg_seeds.as_slice());
                    if dist > furthest_obs.1 {
                        furthest_obs = (ix, dist);
                    }
                }
            }
            furthest_obs.0
        };
        let mut sc = sample.clone().nth(rand_obs).unwrap();
        let sb = sc.borrow();
        seeds.push((rand_obs, DVector::from_iterator(sb.len(), sb.iter().cloned() )));
    }*/

    // Take first seeds as random. Then take next seed to be the observation that
    // is furthest from all the previously-encountered seeds (with the highest average
    // distance to all seeds), increasing the chance of taking seeds that are far apart.
    for c in 0..settings.n_cluster {
        let rand_obs = if c == 0 {
            (0..n).choose(&mut rand::thread_rng()).unwrap()
        } else {
            let mut furthest_obs = (0, 0.0);
            for (ix, obs) in sample.clone().enumerate() {
                let obs_is_seed = seeds.iter().any(|(s_ix, _)| *s_ix == ix );
                if !obs_is_seed {
                    let mut avg_dist = 0.0;
                    for s in seeds.iter() {
                        avg_dist += euclidian(obs.borrow(), s.1.as_slice());
                    }
                    avg_dist /= seeds.len() as f64;
                    if avg_dist > furthest_obs.1 {
                        furthest_obs = (ix, avg_dist);
                    }
                    // let furthest_seed_form_obs = furthest(seeds.iter().map(|(_, s)| s.as_slice() ), obs);
                    // let dist = euclidian(obs.borrow(), seeds.iter().nth(furthest_seed_form_obs).unwrap().1.as_slice() );
                    // if dist > furthest_obs.1 {
                    //    furthest_obs = (ix, dist);
                    //}
                }
            }
            furthest_obs.0
        };
        let mut sc = sample.clone().nth(rand_obs).unwrap();
        let sb = sc.borrow();
        seeds.push((rand_obs, DVector::from_iterator(sb.len(), sb.iter().cloned() )));
    }

    let mut allocations = Vec::with_capacity(n);
    for (ix, obs) in sample.enumerate() {

        // For the few observations that were chosen as seeds, its allocation is the index at seed vector.
        if let Some(c) = seeds.iter().position(|(si, _)| ix == *si ) {
            allocations.push(c);

        // For the remaining observations, take the index at the seed vector that is closest
        // to this observation.
        } else {
            allocations.push(closest( seeds.iter().map(|(_, v)| v.as_slice() ), obs.borrow() ));
        }
    }

    allocations
}

fn valid_allocations(allocations : &[usize], settings : &KMeansSettings) -> bool {
    allocations.iter().all(|a| (0..settings.n_cluster).any(|c| c==*a ) )
}

impl Estimator for KMeans {

    type Settings = KMeansSettings;

    type Error = KMeansError;

    fn estimate(
        sample : impl Iterator<Item=impl Borrow<[f64]>> + Clone,
        mut settings : Self::Settings
    ) -> Result<Self, Self::Error> {

        let n = sample.clone().count();
        let obs_dim = sample.clone().next().unwrap().borrow().len();

        // TODO initialize with dendrogram of close observations, instead of random allocations.
        // Staring with a dendrogram favors identifiability, by exploring a space where
        // observations are already close to one another, instead of being in a space where
        // all possible optimal configurations are not yet explored.
        let mut allocations : Vec<usize> = if let Some(allocs) = settings.allocations.take() {
            allocs
        } else {
            seed_allocations(&settings, sample.clone(), n)
        };

        // let mut allocations : Vec<usize> = random_allocations(&settings, n);

        assert!(valid_allocations(&allocations[..], &settings));

        let mut clusters = HashMap::new();
        for ix in 0..settings.n_cluster {
            clusters.insert(ix, ClusterInfo { sum : DVector::zeros(obs_dim), count : 0 });
        }

        let mut means : Vec<DVector<f64>> = (0..settings.n_cluster)
            .map(|_| DVector::zeros(obs_dim) )
            .collect();

        let mut n_iter = 0;
        for _ in 0..settings.max_iter {
            n_iter += 1;
            update_means(&mut means[..], &mut clusters, sample.clone(), &allocations[..], &settings);
            let n_reallocated = update_allocations(&mut allocations[..], &means[..], sample.clone());
            if n_reallocated == 0 {
                break;
            }
        }

        assert!(valid_allocations(&allocations[..], &settings));

        Ok(Self {
            allocations,
            means,
            n_iter
        })
    }

}

/*#[test]
fn kmeans() {

    use crate::{prob::{Normal, Prior}, fit::{cluster::KMeans, cluster::KMeansSettings, Estimator}};
    use rand::thread_rng;
    use rand_distr::Distribution;
    use std::iter::FromIterator;

    let n1 = Normal::prior(0.0, Some(1.));
    let n2 = Normal::prior(5.0, Some(1.));
    let n3 = Normal::prior(15.0, Some(1.));

    let data = Vec::from_iter((0..300).map(
        |i| if i % 3 == 0 {
            n1.sample(&mut thread_rng())
        } else {
            if i % 2 == 0 {
                n2.sample(&mut thread_rng())
            } else {
                n3.sample(&mut thread_rng())
            }
        }
    ));

    let km = KMeans::estimate(
        data.iter().map(|d| [*d] ),
        KMeansSettings { n_cluster : 3, max_iter : 1000, allocations : None }
    ).unwrap();

    println!("{}", km);

}*/


