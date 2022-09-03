use nalgebra::*;
use nalgebra::storage::*;
use std::cmp::Ordering;
use super::*;
use std::iter::FromIterator;
use std::fmt;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;
use crate::prob::{Normal, Exponential, STANDARD_NORMAL};
use num_traits::AsPrimitive;
use crate::calc::Variate;
use std::ops::{Add, AddAssign, Div, Mul};
use std::ops::Range;
use num_traits::Zero;
use crate::calc::rank::Rank;
use num_traits::Float;

pub struct Partition {
    pub mode : usize,
    pub first : usize,
    pub last : usize
}

use petgraph::algo::simple_paths::all_simple_paths;
use petgraph::DiGraph;
use petgraph::NodeIndex;
use petgraph::visit::{DfsEvent, Control};

/// Finds a histogram partition over an unknown number of modes (assumed to be
/// within a user-provided interval) that minimizes the difference between the
/// marginal histogram entropy and the conditional local histograms cross-entropies
/// wrt. the marginal histogram. If the histogram were to be a mixture distribution,
/// with a perfect analytical description like m(x) = \sum_k p_k(x)theta_k,
/// then weighted cross entropies should be equal to the marginal cross-entropy.
/// Since this most likely won't be the case if we define our functions as a partition
/// over the variable domain, we settle for the partition that minimizes the error
/// to this situation. Assumes the partition intervals are symmetrical around histogram
/// modes to keep the search space to a reasonable size, although this assumption could
/// be relaxed to allow skewed distributions.
/// Arguments
/// p : A probability histogram.
/// min_mode, max_mode: Mode search space
///
pub fn min_partial_entropy_partition(
    probs : &[f32],
    min_mode : usize,
    max_mode : usize,
    min_range : usize,
    max_range : usize
) -> Vec<Partition> {

    // Holds as many local maxima as are possible by array size / min_range.
    let max_intervals = probs.len() / min_range;

    let mut max_merge_sz = 1;
    while max_merge_sz*min_range <= max_range {
        max_merge_sz += 1;
    }

    let mut local_maxima = Vec::new();
    for _ in 0..max_intervals {
        let past_sz = local_maxima.len();
        super::single_interval_search(
            p
            &mut local_maxima,
            min_range / 2,
            local_mode,
            false
        );
        if local_maxima.len() == past_sz {
            break;
        }
    }

    // Make intervals contiguous.
    let n_intervals = local_maxima.len();
    for i in 1..n_intervals {
        let half_dist = (local_maxima[i].0.start - local_maxima[i-1].0.end) / 2;
        if half_dist > 1 {
            local_maxima[i-1].0.end += half_dist;
            local_maxima[i].0.start = local_maxima[i-1].0.end;
        }
    }

    let (valid_merge, merge_ranges) = determine_valid_merges(n_intervals, min_mode, max_mode);
    let acc_probs : Vec<_> = crate::calc::cumulative_sum(p.iter().copied()).collect();
    let acc_marg_entropy = crate::calc::cumulative_entropy(acc_probs.iter().copied());

    let marg_entropy = acc_marg_entropy.last().unwrap();
    let mut smallest_entropy = f32::MAX;
    let mut best_partition = Vec::new();

    // Now, visit histogram to find the weighted cross-entropy that most closely matches
    // the marginal cross-entropy.
    let mut hist_ranges = Vec::new();
    for merge_range in merge_ranges {
        determine_hist_ranges(&mut hist_ranges, &valid_merges[..], &merge_ranges[..]);

        let mut total_cond_entropy = 0.0;
        for r in &hist_ranges {
            let p = acc_probs[r.end-1] - acc_probs[r.start];

            // Gets section of marginal entropy corresponding to this conditional
            let partial_entropy = acc_marg_entropy[r.end-1] - acc_marg_entropy[r.start];

            // The actual conditional entropy is a simple function of the partial marginal entropy
            let cond_entropy = partial_entropy / p - p.ln();

            total_cond_entropy += cond_entropy;
        }

        if total_cond_entropy < smallest_entropy {
            smallest_entropy = total_cond_entropy;
            best_partition.clear();
            for r in &hist_ranges {
                let mode = local_maxima[r].iter()
                    .max_by(|a, b| a.1.val.partial_cmp(&b.1.val).unwrap_or(Ordering::Equal) )
                    .unwrap();
                let (fst, lst) = (local_maxima[r].first().unwrap().0, local_maxima[r].last().unwrap());
                let part = Partition {
                    mode,
                    first : fst.0.start,
                    last : lst.0.end
                };
                best_partition.push(part);
            }
        }
    }
    partitions
}

// This could be determined beforehand and saved to a JSON, for example,
// mapping pairs n_intervals:min_mode:maxmode -> resulting graph, or required
// to be computed only once by the user, that would pass this graph into
// min_max_cross_entropy_partition (perhaps wrapped in a config-like structure).
fn determine_valid_merges(
    n_intervals : usize,
    min_mode : usize,
    max_mode : usize
) -> (Vec<bool>, Vec<Range<usize>>) {

    // Graph that dictates if the interval (index of local_maxima as node)
    // should merge with the interval to the right (edge=true indicates a merge).
    let mut merge_graph = DiGraph::<usize, bool>::new_directed();
    let fst = merge_graph.add_node(0);
    grow(&mut merge_graph, (fst, false), 0, n_intervals-1);
    grow(&mut merge_graph, (fst, true), 0, n_intervals-1);

    // Final merge accumulators
    let mut valid_merges : Vec<bool> = Vec::new();
    let mut curr_merge : Vec<bool> = Vec::new();
    let mut merge_ranges : Vec<Range<usize>> = Vec::new();

    // Graph state.
    let mut prev_ix = NodeIndex::from(0);
    let mut n_contiguous = 1;
    let mut n_merges = 1;

    for n_modes in min_mode..=max_mode {
        // Filter graph paths with #false - 1 == n_modes AND max(contiguous trues) <= max_merge_sz.

        petgraph::visit::depth_first_search(&merge_grah, Some(fst), |event| {
            match event {
                DfsEvent::Discover(ix, _) => {

                    if merge_grah[ix] == 0 {

                        // Starting a new path

                        assert!(curr_merge.len() == 0);

                        n_merges = 1;
                        n_contiguous = 1;
                        prev_ix = ix;
                        Control::Continue
                    } else if merge_grah[ix] == n_intervals-1 {

                        // Ending last path

                        let len_before = valid_merges.len();
                        valid_merges.extend(curr_merge.drain(..));
                        merge_ranges.push(Range { start : len_before, end : valid_merges.len() } );
                        prev_ix = ix;
                        Control::Continue
                    } else {

                        // Decide if merge path should be considered any further.

                        let e = merge_grah.find_edge(prev_ix, ix).unwrap();
                        let should_merge = merge_grah[&e];

                        if should_merge {
                            n_contiguous += 1;
                        } else {
                            n_contiguous = 1;
                            n_merges += 1;
                        }

                        if n_merges <= n_modes && n_contiguous <= max_merge_sz {
                            curr_merge.push(should_merge);
                            prev_ix = ix;
                            Control::Continue
                        } else {
                            curr_merge.clear();
                            prev_ix = ix;
                            n_contiguous = 1;
                            n_merges = 1;
                            Control::Prune
                        }
                    }
                },
                _ => {
                    Control::Continue
                }
            }
        });
    }

    (valid_merges, merge_ranges)
}

fn determine_hist_ranges(
    hist_ranges : &mut Vec<Range<usize>>,
    valid_merges : &[bool],
    merge_ranges : &[Range<usize>]
) {
    hist_ranges.clear();
    let mut curr_range = Range { start : 0, end : 1 };
    for should_merge in merge_range {
        if should_merge {
            curr_range.end += 1;
        } else {
            hist_ranges.push(curr_range);
            curr_range = Range { start : i, end : i+1 };
        }
    }
}

fn grow(merge_graph : &mut DiGraph::<usize, bool>, prev : (NodeIndex, bool), curr : usize, max : usize) {
    let new = merge_graph.add_node(curr);
    graph.add_edge(prev.0, new, prev.1);
    if curr < max {
        grow(merge_graph, (new, false), curr + 1, max);
        grow(merge_graph, (new, true), curr + 1, max);
    }
}

fn local_mode(probs : &[f32], range : Range<usize>, step : usize) -> Option<(Range<usize>, Mode)> {
    let local_mode = crate::calc::frequency::mode(&probs[range]);
    let pos = local_mode.pos + range.start;
    let mut start = pos.saturating_sub(step);

    // Extend first and last intervals.
    if start < step {
        start = 0;
    }
    let mut end = (pos + step).min(probs.len());
    if probs.len() - end < step {
        end = probs.len();
    }

    Some((Range { start, end }, Mode { pos, val : local_mode.val }))
}

// mean += bin*prob where prob = count/total (but divide by total at the end)
pub fn mean_for_hist<B>(h : &[B]) -> B
where
    B : AsPrimitive<f32> + Copy + Zero + 'static + AddAssign + Div<Output=B> + Mul<Output=B>,
    usize : AsPrimitive<B>,
{
    let mut weighted_sum = B::zero();
    let mut total = B::zero();
    for (bin, v) in h.iter().enumerate() {
        weighted_sum += bin.as_() * *v;
        total += *v;
    }
    weighted_sum / total
}

pub fn median_for_accum_hist<B>(h : &[B]) -> Rank<B>
where
    B : AsPrimitive<f32> + Copy + 'static,
    f32 : Div<Output=f32>
{
    quantile_for_accum_hist(h, 0.5)
}

pub fn quantile_for_accum_hist<B>(h : &[B], q : f32) -> Rank<B>
where
    B : AsPrimitive<f32> + Copy + 'static,
    f32 : Div<Output=f32>
{
    let max_val : f32 = h.last().copied().unwrap().as_();
    let pos = h.partition_point(|a| a.as_() / max_val <= q );
    Rank { pos, val : h[pos] }
}

/*// Suppose you calculate means and vars separately at each dimension of a multivariate
// quantity. The join bivariate means and variances will be located at the local
// maximums of the products of the corresponging normals.
pub fn join_univariate(means : &[f64], vars : &[f64]) {
    let min_z = 3.0;
    STANDARD_NORMAL *
}*/

// Extract normal distributions by:
// (1) Identify mode
// (2) Expand to the left and right, creating a new normal model with the extreme samples.
// Iterate over those steps refining the normal model as long as the new normal model either:
// (a) don't contain values that are too extreme (z_thresh)
// (b) The new samples aren't too unlikely under the old model (log_odds_thresh)
// corresponding normal z-score.
// Repeat the above steps recursively until no more normals can be found.
// This can be called at each separate dimension of a multivariate quantity to
// jumpstart an expectation-maximization algorithm with independent gaussians.
pub fn normals_at_hist<U>(
    hist : &[U],
    z_thresh : Option<f32>,
    log_odds_thresh : Option<f64>,
    min_prob : f32
) -> Vec<Normal>
where
    U : PartialOrd + AsPrimitive<f32> + Add<Output=U> + std::iter::Sum,
    f64 : From<U>
{

    // TODO use default parameters here instead.
    assert!(log_odds_thresh.is_some() || z_thresh.is_some());

    // assert!(z_thresh >= 0.0);
    assert!(min_prob >= 0.0 && min_prob <= 1.0);

    let sum : f32 = hist.iter().copied().sum::<U>().as_();
    let probs : Vec<_> = hist.iter().map(|b| { let rel : f32 = b.as_(); rel / sum }).collect();
    let cumul_probs : Vec<_> = crate::calc::running::cumulative_sum(probs.iter().copied()).collect();
    let domain : Vec<_> = hist.iter().map(|b| { let rel : f32 = b.as_(); rel / sum }).collect();

    let mut normals = Vec::new();
    let mut new_ranges = Vec::new();
    let begin = hist.as_ptr() as usize;
    let mut ranges = Vec::new();
    let mut normal_is_likely = true;
    while normal_is_likely {
        crate::calc::iter_disjoint(hist, &ranges[..], |local_hist| {
            let offset = local_hist.as_ptr() as usize - begin;
            let max = local_hist.iter().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal) ).unwrap().0;
            let mut low = max.saturating_sub(1);
            let mut high = (max + 1).min(local_hist.len()-1);
            let mut mean = probs[offset+low]*(low as f32) +
                probs[offset+max]*(max as f32) +
                probs[offset+high]*(high as f32);
            let mut var = probs[offset+low]*(low as f32).powf(2.0) +
                probs[offset+max]*(max as f32).powf(2.0) +
                probs[offset+high]*(high as f32).powf(2.0);

            let mut stddev = var.sqrt();
            let mut z_low = (low as f32).standardize(&mean, &stddev);
            let mut z_high = (high as f32).standardize(&mean, &stddev);

            // Count the log_probability N times, since the histogram has N samples
            let mut lp_low = STANDARD_NORMAL.log_probability(z_low as f64)*(f64::from(hist[offset+low]));
            let mut lp_high = STANDARD_NORMAL.log_probability(z_high as f64)*(f64::from(hist[offset+high]));

            let mut expanded_low = true;
            let mut expanded_high = true;
            let mut z_valid = true;
            let mut lp_valid = true;
            while (expanded_low || expanded_high) && z_valid && lp_valid  {

                if low > 0 {
                    low -= 1;
                    mean += probs[offset+low]*(low as f32);
                    var += probs[offset+low]*(low as f32).powf(2.0);
                } else {
                    expanded_low = false;
                }

                if high < local_hist.len()-1 {
                    high += 1;
                    mean += probs[offset+high]*(high as f32);
                    var += probs[offset+high]*(high as f32).powf(2.0);
                } else {
                    expanded_high = false;
                }

                stddev = var.sqrt();
                z_low = (low as f32).standardize(&mean, &stddev);
                z_high = (high as f32).standardize(&mean, &stddev);
                z_valid = if let Some(z) = z_thresh {
                    z_low >= (-1.0)*z && z_high <= z
                } else {
                    true
                };

                let new_lp_low = STANDARD_NORMAL.log_probability(z_low as f64)*(f64::from(hist[offset+low]));
                let new_lp_high = STANDARD_NORMAL.log_probability(z_high as f64)*(f64::from(hist[offset+high]));
                lp_valid = if let Some(lo_thresh) = log_odds_thresh {
                    lp_low - new_lp_low >= lo_thresh && lp_high - new_lp_high >= lo_thresh
                } else {
                    true
                };
                lp_low = new_lp_low;
                lp_high = new_lp_high;
            }

            normals.push(Normal::new(offset as f64 + mean as f64, var as f64));
            new_ranges.push(Range { start : offset+low, end : offset+high });
        });
        ranges.extend(new_ranges.drain(..));

        // Take the least likely normal
        let rare_range = ranges.iter().min_by(|a, b| {
            (cumul_probs[a.end]-cumul_probs[a.start]).partial_cmp(&(cumul_probs[b.end]-cumul_probs[b.start]))
            .unwrap_or(Ordering::Equal)
        }).unwrap();
        normal_is_likely = (cumul_probs[rare_range.end]-cumul_probs[rare_range.start]) >= min_prob;
    }
    normals
}

/// Represents the empirical cumulative distribution function (ECDF) of a sample.
pub struct Cumulative {

    rel_cumul : DVector<f64>,

    abs_cumul : DVector<f64>

}

impl Cumulative {

    pub fn calculate<'a>(sample : impl Iterator<Item=&'a f64>) -> Self {
        let mut s = Vec::from_iter(sample.cloned());
        s.sort_unstable_by(|s1, s2| s1.partial_cmp(s2).unwrap_or(Ordering::Equal) );
        let n = s.len();
        let (mut rel_cumul, mut abs_cumul) = (DVector::zeros(n), DVector::zeros(n));
        let s_min : f64 = s[0];
        let s_max : f64 = s[n - 1];
        let ampl = (s_max - s_min).abs();

        rel_cumul[0] = s[0] / ampl;
        abs_cumul[0] = s[0];
        for i in 1..n {
            rel_cumul[i] = rel_cumul[i-1] + s[i] / ampl;
            abs_cumul[i] = abs_cumul[i-1] + s[i];
        }

        Self { rel_cumul, abs_cumul }
    }

}

#[derive(Debug)]
pub struct Histogram {

    min : f64,

    max : f64,

    // Size of fragmented intervals
    intv : f64,

    // Full interval to be represented
    ampl : f64,

    // Number of samples
    n : usize,

    n_bins : usize,

    // Stores intervals of same size, with the interval order as key and count of ocurrences
    // as values. Intervals without an element count are missing entries.
    bins : BTreeMap<u64, u64>
}

#[derive(Serialize, Deserialize)]
pub struct Bin {
    pub low : f64,
    pub high : f64,
    pub count : u64,
    pub prop : f64
}

impl Histogram {

    pub fn cumulative(&self) -> Vec<u32> {
        let mut cumul = Vec::new();
        let mut s : u32 = 0;
        for (_, v) in self.bins.iter() {
            s += *v as u32;
            cumul.push(s);
        }
        cumul
    }

    pub fn calculate<'a>(sample : impl Iterator<Item=&'a f64> + Clone, n_bins : usize) -> Self {
        let (mut min, mut max) = (f64::MAX, f64::MIN);
        let mut n = 0;
        for s in sample.clone() {
            if *s > max {
                max = *s;
            }
            if *s < min {
                min = *s;
            }
            n += 1;
        }

        let ampl = max - min;
        let intv = ampl / n_bins as f64;
        let mut bins = BTreeMap::<u64, u64>::new();
        for s in sample {

            // Retrieves the discrete bin allocation for s as the ordered interval
            let b = ((*s - min) / intv).floor().abs() as u64;
            let b_count = bins.entry(b).or_insert(0);
            *b_count += 1;
        }

        Self { min, max, n, intv, ampl, bins, n_bins }
    }

    /// Returns 0 if bin position is outside bounds or no elements are allocated to it.
    pub fn count(&self, pos : usize) -> u64 {
        *self.bins.get(&(pos as u64)).clone().unwrap_or(&0)
    }

    /// Returns 0 if bin position is outside bounds or no elements are allocated to it.
    pub fn proportion(&self, pos : usize) -> f64 {
        self.bins.get(&(pos as u64)).map(|b| *b as f64 / self.n as f64 ).unwrap_or(0.0)
    }

    pub fn bounds(&self, pos : usize) -> Option<(f64, f64)> {
        let low = self.min + self.intv*(pos as f64);
        let high = low + self.intv;
        Some((low, high))
    }

    pub fn bin(&self, pos : usize) -> Option<Bin> {
        let (low, high) = self.bounds(pos)?;
        let prop = self.proportion(pos);
        let count = self.count(pos);
        Some(Bin{ low, high, prop, count })
    }

    pub fn iter_bins(&self) -> Vec<Bin> {
        (0..self.n_bins).map(|pos : usize| self.bin(pos).unwrap() ).collect()
    }
}

/*/// Structure to represent one-dimensional empirical distributions non-parametrically (Work in progress).
/// One-dimensional histogram, useful for representing univariate marginal distributions
/// of sampled posteriors non-parametrically. Retrieved by indexing a Sample structure.
/// The histogram resolution is a linear function of the sample size. Histograms can be
/// thought of as a non-parametric counterpart to bayes::distr::Normal in the sense that it
/// represents a distribution over an unbounded real variable. But unlike Normals, which are
/// defined only by mean and variance, histograms can represent any empiric univariate distribution.
/// Histogram implements Distribution API.
pub struct Histogram {

    ord_sample : DVector<f64>,

    // homogeneous partition of the variable domain.
    prob_intv : f64,

    // Values of the variable that partition the
    // cumulative sum into homogeneous probability
    // intervals.
    bounds : DVector<f64>,

    mean : f64,

    variance : f64,

    full_interval : f64
}

impl Histogram {

    pub fn calculate<'a>(sample : impl Iterator<Item=&'a f64>) -> Self {
        let s = DVector::from(Vec::from_iter(sample.cloned()));
        Self::calculate_from_vec(&s)
    }

    pub fn calculate_from_vec<S>(sample : &Matrix<f64, Dynamic, U1, S>) -> Self
        where S : Storage<f64, Dynamic, U1>
    {

        // TODO consider using itertools::sorted_by
        assert!(sample.nrows() > 5, "Sample too small to build histogram");
        let mut sample_c = sample.clone_owned();
        let mut sample_vec : Vec<_> = sample_c.data.into();
        sample_vec.sort_unstable_by(|s1, s2| s1.partial_cmp(s2).unwrap_or(Ordering::Equal) );
        let ord_sample = DVector::from_vec(sample_vec);
        let n = ord_sample.nrows();
        let s_min : f64 = ord_sample[0];
        let s_max : f64 = ord_sample[n - 1];
        let n_bins = n / 4;
        let mut acc = s_min + s_max;
        let mut acc_sq = s_min.powf(2.) + s_max.powf(2.);
        let mut bounds = DVector::zeros(n_bins+1);
        bounds[0] = s_min;
        bounds[n_bins] = s_max;
        let mut curr_bin = 0;
        let prob_intv = 1. / n_bins as f64;
        for i in 1..(ord_sample.nrows() - 1) {
            acc += ord_sample[i];
            acc_sq += ord_sample[i].powf(2.);
            //println!("sample:{}", ord_sample[i]);
            //println!("acc: {}", acc);
            if i as f64 / n as f64 > (curr_bin as f64)*prob_intv {
                bounds[curr_bin] = (ord_sample[i-1] + ord_sample[i]) / 2.;
                curr_bin += 1;
            }
        }
        assert!(curr_bin == bounds.nrows() - 1);
        let mean = acc / (n as f64);
        let variance = acc_sq / (n as f64) - mean.powf(2.);
        let full_interval = bounds[bounds.nrows() - 1] - bounds[0];
        Self{ ord_sample, prob_intv, bounds, mean, variance, full_interval }
    }

    pub fn median(&self) -> f64 {
        self.quantile(0.5)
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn var(&self) -> f64 {
        self.variance
    }

    pub fn limits(&self) -> (f64, f64) {
        let n = self.ord_sample.nrows();
        (self.ord_sample[0], self.ord_sample[n-1])
    }

    /// Returns value such that cumulative probability == prob.
    pub fn quantile(&self, prob : f64) -> f64 {
        assert!(prob >= 0.0 && prob <= 1.0, "Invalid probability value");
        let p_ix = ((self.bounds.nrows() as f64) * prob) as usize;
        //(self.bounds[p_ix] + self.bounds[p_ix - 1]) / 2.
        self.bounds[p_ix]
    }

    /// Returns cumulative probability up to the informed value, by starting
    /// the search over the accumulated bounds at the informed value, and returning
    /// the number of iterations performed.
    fn prob_bounded(&self, value : f64, lower_bound : usize) -> (f64, usize) {
        assert!(value <= self.bounds[self.bounds.nrows() - 1], "Value should be < sample maximum");
        let diff_ix = self.bounds.rows(lower_bound, self.bounds.nrows() - lower_bound)
            .iter().position(|b| *b >= value).unwrap();
        //(self.bounds[b_ix] - self.bounds[0]) / self.full_interval
        ((lower_bound+diff_ix) as f64 * self.prob_intv, diff_ix)
    }

    /// Returns cumulative probability up to the informed value, performing a full serch
    /// over the histogram.
    fn prob(&self, value : f64, bound : usize) -> f64 {
        self.prob_bounded(value, 0).0
    }

    /// Returns (bin right bound, probabilities) pair when the full interval is
    /// partitioned into nbin intervals.
    pub fn full(&self, nbins : usize, cumul : bool) -> (DVector<f64>, DVector<f64>) {
        assert!(nbins > 3);
        let interval = self.full_interval / nbins as f64;
        let mut values = DVector::from_iterator(
            nbins,
            (1..(nbins+1)).map(|b| self.bounds[0] + (b as f64)*interval )
        );
        let mut curr_bound = 0;
        let cumul_prob = values.map(|v| {
            let (p,nd) = self.prob_bounded(v, curr_bound);
            curr_bound += nd;
            p
        });
        //println!("prob: {}", cumul_prob);
        if cumul {
            (values, cumul_prob)
        } else {
            let mut prob = DVector::zeros(cumul_prob.nrows());
            prob[0] = cumul_prob[0];
            for i in 1..prob.nrows() {
                prob[i] = cumul_prob[i] - cumul_prob[i-1];
            }
            (values, prob)
        }
    }

    pub fn quartiles(&self) -> (f64, f64) {
        (self.quantile(0.25), self.quantile(0.75))
    }

    pub fn subsample(&self, nbins : usize) -> Histogram {
        let (full, probs) = self.full(nbins, false);
        let n = full.nrows();
        let mut sample = Vec::new();
        for (x, p) in full.iter().zip(probs.iter()) {
            sample.extend((0..(n as f64 * p) as usize).map(|_| x ));
        }
        Self::calculate_from_vec(&DVector::from_vec(sample))
    }

}

impl fmt::Display for Histogram {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (fst, last) = self.limits();
        let (quart_low, quart_high) = self.quartiles();
        let mean = self.mean();
        let med = self.median();
        let var = self.var();
        write!(
            f,
            "Histogram {{ Limits: [{} - {}]; Quartiles: [{} - {}]; Mean: {}; Median: {}; Variance: {} }}",
            fst,
            last,
            quart_low,
            quart_high,
            mean,
            med,
            var
        )
    }

}

pub struct JointCumulative {

}

impl JointCumulative {

}

/// Useful to represent the joint distribution of pairs of non-parametric
/// posterior parameters and to calculate pair-wise statistics for them, such
/// as their covariance and correlation. This can be thought of as a nonparametric
/// counterpart to bayes::prob::MultiNormal of dimensionality 2. Representing distributions
/// of higher-dimensions non-parametrically becomes computationally infeasible for dimensionality
/// greater than two, so you must resort to a parametric formulation in this case; or at least
/// to a mixture. Distributions can be through of as living in the continum of flexibility x tractability:
/// <-- Flexibility         Tractability -->
/// Histograms      Mixtures       Parametric (Expoential-family)
pub struct JointHistogram {
    comm_domain : DVector<f64>,
    joint_prob : DMatrix<f64>,
}

impl JointHistogram {

    pub fn build(a : Histogram, b : Marginal) -> Self {
        /*let n = b.len();
        let mut cols = Vec::new();
        for i in 0..n {
            cols.push(b[i])
        }*/
        //Self{ bottom : a.subsample(n), cols }
        unimplemented!()
    }

    pub fn quantile(pa : f64, pb : f64) -> (f64, f64) {
        /*let qa = self.quantile(pa);
        let ix = self.comm_domain.position(|b| b >= qa).unwrap();
        self.cols[ix].quantile(pb);
        (pa, pb)*/
        unimplemented!()
    }

    pub fn prob(pa : f64, pb : f64) -> f64 {
        /*let x = self.quantile(pa);
        self.bottom.prob(p) **/
        unimplemented!()
    }

}

/// Pair of marginal histograms.
pub struct Marginal {
    bottom : Histogram,
    right : Histogram
}

impl MarginalHistogram {

    pub fn build(a : Histogram, b : Histogram) -> Self {
        Self{ bottom : a, right : b}
    }

}*/

#[test]
fn histogram() {
    let incr = 1.0;
    let data : Vec<_> = (0..10).map(|s| s as f64 * incr).collect();
    // let hist = Histogram::calculate(data.iter());
    // println!("{}", hist);
    // println!("{}", hist.subsample(10));
}


