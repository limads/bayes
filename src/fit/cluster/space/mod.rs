use std::collections::HashMap;
use spade::rtree::RTree;
use crate::*;
use std::cmp::PartialEq;
use spade::HasPosition;
use crate::fit::cluster::Distance;
use crate::fit::cluster::Euclidian;

/// Implements density-based clustering (dbscan algorithm).
pub struct SpatialClustering<T> {
    pub clusters : HashMap<usize, Vec<T>>,
    pub noise : Vec<T>
}

impl<T> SpatialClustering<T> {

    pub fn cluster_linear(pts : &[T], max_dist : f64, min_clust_sz : usize) -> Self
    where
        T : PartialEq + Distance<Euclidian> + Copy + Clone
    {
        let search = LinearSearch { pts, max_dist };
        density_clustering(&search, max_dist, min_clust_sz)
    }

    pub fn cluster_indexed(pts : &[T], max_dist : f64, min_clust_sz : usize) -> Self
    where
        T : PartialEq + Distance<Euclidian> + spade::PointN + Copy + Clone
    {
        let mut tree : RTree<PointRef<'_, T>> = RTree::new();
        pts.iter().enumerate().for_each(|(lin_ix, pt)| { tree.insert(PointRef { lin_ix, pt }); });
        let search = IndexedSearch {
            pts,
            tree,
            max_dist
        };
        density_clustering(&search, max_dist, min_clust_sz)
    }
}

trait SpatialSearch<'a, T> {

    /// Clears the ixs vector and writes only the indices of the held
    /// points that are spatially close to the reference point.
    fn neighborhood(&'a self, ixs : &mut Vec<usize>, ref_pt : usize);

    fn points(&'a self) -> &'a [T];

}

pub struct LinearSearch<'a, T> {
    pts : &'a [T],
    max_dist : f64
}

impl<'a, T> SpatialSearch<'a, T> for LinearSearch<'a, T>
where
    T : PartialEq + Distance<Euclidian> + Copy + Clone
{

    fn neighborhood(&'a self, neigh : &mut Vec<usize>, ref_ix : usize) {
        neigh.clear();
        let ref_pt = &self.pts[ref_ix];
	    neigh.extend(self.pts.iter().cloned()
	        .enumerate()
	        .filter(|(_, pt)| !pt.eq(ref_pt) && ref_pt.distance(pt) < self.max_dist )
	        .map(|(ix, _)| ix )
	    );
    }

    fn points(&'a self) -> &'a [T] {
        &self.pts[..]
    }
}

struct IndexedSearch<'a, T>
where
    PointRef<'a, T>: spade::HasPosition
{
    pts : &'a [T],
    tree : RTree<PointRef<'a, T>>,
    max_dist : f64
}

/// This structure is required because we still want to
/// keep a linear index to the elements being aggregated,
/// so instead of storing the elements in the RTree, we store
/// this lightweight structue that carries a reference and a linear
/// index. We implement spade::HasPosition for it by just returning
/// a copy of the desired element (perhaps instead of cloning, we
/// just make type Point = &T).
#[derive(Clone, Copy, Debug)]
struct PointRef<'a, T> {
    pt : &'a T,
    lin_ix : usize
}

impl<'a, T> spade::HasPosition for PointRef<'a, T>
where
    T : Copy + Clone + spade::PointN
{

    type Point = T;

    fn position(&self) -> Self::Point {
        self.pt.clone()
    }

}

impl<'a, T> SpatialSearch<'a, T> for IndexedSearch<'a, T>
where
    T : PartialEq + Distance<Euclidian>,
    PointRef<'a, T>: spade::HasPosition
{

    fn neighborhood(&'a self, neigh : &mut Vec<usize>, ref_ix : usize) {
        neigh.clear();
        let ref_pt = &self.pts[ref_ix];
        let ref_pt_t = PointRef { pt : ref_pt, lin_ix : ref_ix };

        // TODO the element itself will be returned by this query, so we must actually check
        // for N+1 nearest elements in this case.
        neigh.extend(self.tree.nearest_neighbor_iterator(&ref_pt_t.position())
            .take_while(|pt| ref_pt.distance(&pt.pt) < self.max_dist )
            .map(|pt| pt.lin_ix )
        );
    }

    fn points(&'a self) -> &'a [T] {
        &self.pts[..]
    }
}

fn expand_local_neighborhood<'a, T>(
    labels : &mut [Assignment],
    search : &'a impl SpatialSearch<'a, T>,
    max_dist : f64,
    min_cluster_sz : usize,
    curr_clust : usize,
    n_ix : usize
) {
    let mut inner_neigh = Vec::new();
    match labels[n_ix] {
	    Assignment::Unvisited => {

	        /*if let Some(tree) = tree {
                local_neighborhood_indexed(&mut inner_neigh, pts[n_ix], pts, tree, max_dist)
	        } else {
	            local_neighborhood_linear(&mut inner_neigh, pts[n_ix], pts, max_dist)
	        };*/
	        search.neighborhood(&mut inner_neigh, n_ix);
	        if inner_neigh.len() > min_cluster_sz {
	            labels[n_ix] = Assignment::Core(curr_clust);
		        for inner_n_ix in inner_neigh {
			        expand_local_neighborhood(labels, search, max_dist, min_cluster_sz, curr_clust, inner_n_ix);
		        }
	        } else {
	            labels[n_ix] = Assignment::Border(curr_clust);
	        }
	    },
	    _ => {
            // labels[n_ix] = Assignment::Border(curr_clust);
	    }
	}
}

enum Assignment {
    Unvisited,
	Noise,
	Core(usize),
	Border(usize),
}

// dbscan implementation (https://en.wikipedia.org/wiki/DBSCAN)
// Receives minimum distance to define core points and minimum number
// of points to determine a cluster allocation. Finding predominantly
// vertical or predominanlty horizontal edges can be done by giving
// asymetric weghts w1, w2 \in [0,1] w1+w2=1 to the vertical and horizontal
// coordinates, effectively "compressing" the points closer either in the
// vertical or horizontal dimension. Returns custers and noise vector.
// This implementation is generic wrt. the spatial seach strategy employed.
fn density_clustering<'a, T>(
    search : &'a impl SpatialSearch<'a, T>,
    max_dist : f64,
    min_cluster_sz : usize
) -> SpatialClustering<T>
where
    T : PartialEq + Distance<Euclidian> + Copy + Clone + 'a
{

	// Point q is **directly reachable** from point p if it is within distance epsilon
	// of core point p.

	// A point q is **indirectly reachable** from p is there is a path of core points where
	// p_i+1 is directly reachable from p_i. q does not need to be a core points,
	// but all other p_i points must.

	// All points not reachable are outliers or noise.

    /*let tree = if use_indexing {

        Some(rtree)
    } else {
        None
    };*/

	let mut labels : Vec<Assignment> = (0..search.points().len()).map(|_| Assignment::Unvisited ).collect();
	let mut curr_clust = 0;
    let mut neigh = Vec::new();

	// (1) For each point pt:
	for (ref_ix, ref_pt) in search.points().iter().enumerate() {

        // Ignore points that might have been classified at previous iterations
	    match labels[ref_ix] {
	        Assignment::Border(_) | Assignment::Core(_) | Assignment::Noise => continue,
	        _ => { }
	    }

		/*if let Some(tree) = tree.as_ref() {
            local_neighborhood_indexed(&mut neigh, *ref_pt, pts, tree, max_dist)
		} else {
		    local_neighborhood_linear(&mut neigh, *ref_pt, pts, max_dist)
		};*/
		search.neighborhood(&mut neigh, ref_ix);

		// If at least min_pts points are within distance epsilon of a point p,
		// it is a core point.
		if neigh.len() >= min_cluster_sz {
			labels[ref_ix] = Assignment::Core(curr_clust);

			// Search for directly-reachable points to this core point still classified as noise
			for n_ix in &neigh {
				expand_local_neighborhood(&mut labels, search, max_dist, min_cluster_sz, curr_clust, *n_ix);
			}
			curr_clust += 1;
		} else {
		    labels[ref_ix] = Assignment::Noise;
		}
	}

	let mut clusters = HashMap::new();
	let mut noise = Vec::new();
	for (lbl_ix, lbl) in labels.iter().enumerate() {
		match lbl {
			Assignment::Core(ix) | Assignment::Border(ix) => {
				clusters.entry(*ix).or_insert(Vec::new()).push(search.points()[lbl_ix]);
			},
			Assignment::Noise => {
                noise.push(search.points()[lbl_ix]);
			},
			Assignment::Unvisited => { panic!("Unvisited point"); }
		}
	}

	SpatialClustering {
	    clusters,
	    noise
	}
}

/*/// Returns index at parent vector and point.
fn local_neighborhood_linear(
    neigh : &mut Vec<usize>,
    ref_pt : (usize, usize),
    pts : &[(usize, usize)],
    max_dist : f64
) {

}

fn local_neighborhood_indexed(
    neigh : &mut Vec<usize>,
    ref_pt : (usize, usize),
    pts : &[(usize, usize)],
    tree : &RTree<PointRef<'_>>,
    max_dist : f64
) {

}*/

