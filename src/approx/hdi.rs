use std;
use std::ops::Add;
use num_traits::Zero;
use num_traits::sign::Unsigned;
use num_traits::cast::{NumCast, ToPrimitive};
use std::ops::AddAssign;
use std::cmp::{Ord, Eq};
use std::ops::Sub;
use std::cmp::{PartialOrd, Ordering};
use std::ops::Range;
use rand_distr::Distribution;

pub struct HDI {
    low : f64,
    high : f64,
}

#[test]
fn test_hdi() {
    let n1 = crate::prob::Normal::new(10.0, 5.0);
    let n2 = crate::prob::Normal::new(0.0, 1.0);
    let mut s = Vec::new();
    for i in 0..1000 {
        s.push(n1.sample(&mut rand::thread_rng()));
        s.push(n2.sample(&mut rand::thread_rng()));
    }
    let hist = crate::approx::Histogram::calculate(s.iter(), 100);
    let cumul = hist.cumulative();
    println!("{:?}", cumul);
    println!("{:?}", hdis(&cumul[..], 4, 4, 2) );
}

// To build Gaussian-like shapes, we can use a weight
// that applies more heavily to the center-most samples.
// Then we calculate k differences within the interval,
// weighted by their distance to the center.

// The resulting vector with size num_intervals
// is sorted in decreasing order of the interval densities.
pub fn hdis(accum_hist : &[u32], step : usize, min_width : usize, num_intervals : usize) -> Vec<Range<usize>> {
    let mut hdis = Vec::new();

    // Populate the hdis vec with the highest density within each disjoint range interval of the previous vector.
    // Iterate recursively until all the remaining ranges are smaller than the required step.
    // Assume hdi vector is sorted by interval position.
    super::multi_interval_search(accum_hist, &mut hdis, step, highest_density_at_range, true);

    hdis.retain(|hdi| hdi.0.end - hdi.0.start >= min_width );
    hdis.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal) );
    hdis.drain(..).map(|hdi| hdi.0 ).take(num_intervals).collect()
}

// Exhaustively search the cumulative histogram for the smallest region with highest mass.
// The optimization varies two parameters, the start and range of the region (which
// can be equivalently represented by start and scale). This ignores previous regions
// at the prev parameter, which allows for localization of different modes by calling it
// recursively starting with empty prev range.
pub fn highest_density_at_range(
    accum_hist : &[u32],
    range : Range<usize>,
    step : usize
) -> Option<(Range<usize>, f32)> {

    if range.end - range.start < step {
        return None;
    }

    let local_hist = &accum_hist[range.clone()];
    let mut hdi_start = 0;
    let mut hdi_stop = local_hist.len();
    let mut highest_dens = 0.0;
    for stop in (step..local_hist.len()).step_by(step) {
        for start in (0..stop).step_by(step).rev() {
            let mass = (accum_hist[stop] - accum_hist[start]) as f32;

            // Those f32s can be cached in a static [256] array, and stop - start used as the index.
            let intv = (stop - start) as f32;

            let dens = mass / intv;

            if dens > highest_dens {
                highest_dens = dens;
                hdi_start = start;
                hdi_stop = stop;
            }
        }
    }

    if highest_dens > 0.0 {
        Some((Range { start : range.start + hdi_start, end : range.start + hdi_stop }, highest_dens))
    } else {
        None
    }
}
