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
use crate::calc::running::*;
use crate::calc::*;
use crate::calc::count::Mode;
use itertools::Itertools;

#[derive(Debug, Clone, Copy)]
pub struct Partition {
    pub mode : usize,
    pub first : usize,
    pub last : usize
}

use petgraph::algo::simple_paths::all_simple_paths;
use petgraph::graph::DiGraph;
use petgraph::graph::NodeIndex;
use petgraph::visit::{DfsEvent, Control};

#[derive(Debug, Clone, Copy)]
pub enum DensityCriteria {

    // Sum over probable regions until the target probability is reached.
    // Probable regions are those with probability satisfying min_local_prob.
    LocalRegion { min_local_prob : f64 },
    
    // Alternatively, iterate over combinations over all disjoint intervals
    // (not only those satisfying min_local_prob), filter those combinations
    // satisfying p >= min_global_prob, and search over the one covering the
    // smallest area. The combinations have size [min_modes, max_modes]
    SmallestArea { min_modes : usize, max_modes : usize } 
}

// Partitions that are closer than spacing and have center mode that is
// higher than both the left and right modes are aggregated.
fn aggregate_modes_symmetrically(
    partitions : &mut Vec<Partition>, 
    probs : &[f64], 
    spacing : usize
) {

}

// Simply aggregate all modes that are closer than spacing (smaller
// modes are aggregated to larger modes).
fn aggregate_modes_asymetrically(
    partitions : &mut Vec<Partition>, 
    probs : &[f64], 
    spacing : usize
) {

}

fn merge_intervals(local_maxima : &mut Vec<(Range<usize>, Mode<f64>)>, i : usize) {
    local_maxima[i].0.end = local_maxima[i+1].0.end;
    if local_maxima[i+1].1.val > local_maxima[i].1.val {
        local_maxima[i].1.val = local_maxima[i+1].1.val;
        local_maxima[i].1.pos = local_maxima[i+1].1.pos;
    }
    local_maxima.remove(i+1);
}

/*
Finds the set of smallest disjoint areas such that each area contains a local mode,
its probability is above or equal to local_probability, and the sum of the probabilities over
the disjoint areas is above or equal to global_probability. If the density criteria smallest_area
is chosen, modes will get merged together, but can be separated if the function is called 
recursively by normalizing the probs over the region of the returned partition.
*/
pub fn highest_density_partition(
    probs : &[f64],
    min_interval : usize,
    step_increment : usize,
    min_global_prob : f64,
    min_mode_ratio : f64,
    crit : DensityCriteria,
    recursive : bool
) -> Vec<Partition> {
    assert!(min_global_prob >= 0.0 && min_global_prob <= 1.0);
    assert!(min_global_prob >= 0.0 && min_global_prob <= 1.0);
    assert!(min_mode_ratio >= 0.0 && min_mode_ratio <= 1.0);
    let mut local_maxima = Vec::new();
    let mut max_intervals = probs.len() / min_interval;
    
    // This would leave regions with far away modes unexplored.
    // if let DensityCriteria::SmallestArea { max_modes, .. } = crit {
    //    max_intervals = max_modes;
    // }

    for _ in 0..max_intervals {
        super::single_interval_search(
            probs,
            &mut local_maxima,
            min_interval,
            local_mode,
            false
        );
    }
    
    // println!("Global prob: {}", min_global_prob);
    let max_mode = local_maxima.iter()
        .max_by(|a, b| a.1.val.partial_cmp(&b.1.val).unwrap_or(Ordering::Equal) 
    ).unwrap().1.val;
    local_maxima.retain(|(_, mode)| mode.val >= min_mode_ratio*max_mode );
    
    let acc_probs : Vec<_> = cumulative_sum(probs.iter().copied())
        .collect();
    // println!("{:?}", probs);
    // println!("{:?}", acc_probs);
    assert!(acc_probs.len() == probs.len());
    let mut local_probs : Vec<f64> = (0..local_maxima.len())
        .map(|_| 0.0 ).collect();
    // println!("local maxima = {:?}", local_maxima);
    
    // Holds indices of a subset of local_probs/local_maxima matching 
    // p >= local_prob (if min_local_prob is informed).
    let mut probable_regions : Vec<usize> = Vec::new();
    
    // Holds sets of findices of local_probs/local_maxima (and the ranges) 
    // matching p >= local_prob
    // (if min_local_prob is not informed).
    // let mut matching_regions : Vec<usize> = Vec::new();
    // let mut matching_ranges : Vec<Range<usize>> = Vec::new();
    
    let mut indices : Vec<usize> = Vec::new();
    let mut best_regions : Vec<usize> = Vec::new();
    
    let mut total_area = 0;
    
    let mut parts = Vec::new();
    
    /* Search in **parallel** the possibilities of having n=1..n=K modes. Then iterate
    over combinations to find the one with the smallest area. */
    
    loop {
    
        // Expand intervals
        for i in 1..(local_maxima.len()-1) {
        
            if local_maxima[i].1.val > local_maxima[i-1].1.val && 
                local_maxima[i].0.start > step_increment 
            {
                local_maxima[i].0.start -= step_increment;
            } else if local_maxima[i-1].0.end < probs.len() - step_increment {
                local_maxima[i-1].0.end += step_increment;
            }
            
            if local_maxima[i].1.val > local_maxima[i+1].1.val && 
                local_maxima[i].0.end < probs.len() - step_increment 
            {
                local_maxima[i].0.end += step_increment;
            } else if local_maxima[i+1].0.start > step_increment {
                local_maxima[i+1].0.start -= step_increment;
            }
        }
        
        // Merge intervals that overlapped.
        for i in (0..(local_maxima.len()-1)).rev() {
            if local_maxima[i].0.end >= local_maxima[i+1].0.start {
                merge_intervals(&mut local_maxima, i);
            }
        }
        // println!("Intervals after merge: {:?}", local_maxima);
        
        calculate_local_probs(&acc_probs, &local_maxima, &mut local_probs);
        // println!("Local probs: {:?}", local_probs);
        
        let area = local_maxima.iter().fold(0, |a, m| a + (m.0.end - m.0.start ) );
        if area == total_area {
            // If area could not be incremented, exit early.
            // println!("Could not achieve global prob.");
            return Vec::new();
        }
        total_area = area;
        
        match crit {
            DensityCriteria::LocalRegion { min_local_prob } => {
                assert!(min_local_prob <= min_global_prob);
                probable_regions.clear();
                let mut p = 0.0;
                for (ix, lp) in local_probs.iter().enumerate() {
                    if *lp >= min_local_prob {
                        p += *lp;
                        probable_regions.push(ix);
                    }
                }
                
                if p >= min_global_prob {
                    for r_ix in &probable_regions {
                        parts.push(Partition { 
                            first : local_maxima[r_ix.clone()].0.start, 
                            last : local_maxima[r_ix.clone()].0.end-1,
                            mode : local_maxima[r_ix.clone()].1.pos
                        }); 
                    }
                    break;
                }
            },
            DensityCriteria::SmallestArea { min_modes, max_modes } => {
    
                if local_maxima.len() < min_modes {
                    return Vec::new();
                }
                
                // If the sum of local areas is smaller than the desired
                // probability, there is no point in doing the more expensive
                // combinatoric examination.
                
                let full_prob = local_probs.iter().fold(0.0, |p, np| p + *np );
                // println!("Evaluating: {:?}", full_prob);
                
                if full_prob < min_global_prob {
                    continue;
                }
                
                indices.clear();
                indices.extend(0..local_maxima.len());
                
                // matching_regions.clear();
                // matching_ranges.clear();
                let mut min_area = None;
                for n_regions in min_modes..(max_modes+1) {
                
                    // Can only combine remaining regions with this
                    // many regions when n >= actual regions.
                    if local_maxima.len() < n_regions {
                        break;
                    }
                    
                    // investigate use of crates permute/permutation to minimize allocations.
                    for comb in indices.iter().combinations(n_regions) {
                        let pr = comb.clone().iter().fold(0.0, |p, ix| {
                            p + local_probs[**ix] 
                        });
                        
                        // Ignore combinations that don't satisfy global prob.
                        if pr >= min_global_prob {
                            let area : usize = comb.iter()
                                .fold(0, |total_area, new_ix| {
                                    let range = &local_maxima[**new_ix].0;
                                    total_area + (range.end - range.start) 
                                });
                            if let Some(ref mut min_area) = min_area {
                                if area < *min_area {
                                    *min_area = area;
                                    best_regions = comb.iter().map(|i| **i ).collect();
                                }
                            } else {
                                min_area = Some(area);
                                best_regions = comb.iter().map(|i| **i ).collect();
                            }
                        } 
                    }
                }
                // println!("Found min area: {:?}", min_area);
                
                if let Some(min_area) = min_area {
                    for r_ix in &best_regions {
                        parts.push(Partition { 
                            first : local_maxima[r_ix.clone()].0.start, 
                            last : local_maxima[r_ix.clone()].0.end-1,
                            mode : local_maxima[r_ix.clone()].1.pos
                        });
                    }
                    assert!(parts.len() <= max_modes);
                    break;
                }
            }
        }
    }
    // println!("Found parts: {:?}", parts);
    
    /* The algorithm could have merged separate modes. Verify if
    partitioning sub-intervals still leave the probability above
    the desired threshold. If the recursive partition finds two
    or more modes, attempt to use them instead. Do nothing otherwise. */
    if recursive {
        for i in (0..parts.len()).rev() {
            probs_for_partitions(&mut local_probs, &acc_probs, &parts);
            if parts[i].last - parts[i].first < 2*min_interval {
                continue;
            }
            let local_probs : Vec<_> = (parts[i].first..=parts[i].last)
                .map(|ix| probs[ix] / local_probs[i] )
                .collect();
            let mut local_parts = highest_density_partition(
                &local_probs[..],
                min_interval,
                step_increment,
                min_global_prob,
                min_mode_ratio,
                crit,
                false
            );
            if local_parts.len() <= 1 {
                continue;
            }
            local_parts.iter_mut().for_each(|p| {
                p.first += parts[i].first; 
                p.last += parts[i].first;
            });
            let mut partition_prob = 0.0;
            for part in &local_parts {
                partition_prob += accumulated_range(
                    &acc_probs[..], 
                    Range { start : part.first, end : part.last+1} 
                );
            }
            let mut prob_compl = 0.0;
            for j in 0..local_probs.len() {
                if j != i {
                    prob_compl += local_probs[j];
                }
            }
            if partition_prob + prob_compl >= min_global_prob {
                parts.remove(i);
                for p in local_parts.iter().rev() {
                    parts.insert(i, p.clone());
                }
            }
        }
    }
    // println!("After recursive: {:?}", parts);
    
    /* The algorithm could have overshoot the desired probability
    due to merges. Trim the found partitions symmetrically until
    the probability is just above the user-desired threshold. */
    let mut n_trimmed = 0;
    loop {
        probs_for_partitions(&mut local_probs, &acc_probs, &parts);
        let p = local_probs.iter().copied().sum::<f64>();
        if p < min_global_prob {
            break;
        }
        n_trimmed = 0;
        for part in parts.iter_mut() {
            if part.last - part.first > min_interval && 
                part.last > part.mode && 
                part.first < part.mode 
            {
                part.first += 1;
                part.last -= 1;
                n_trimmed += 1;
            }
        }
        if n_trimmed == 0 {
            break;
        }
    }
    // println!("After trimming: {:?}", parts);
  
    parts
}

fn probs_for_partitions(local_probs : &mut Vec<f64>, acc_probs : &[f64], parts : &[Partition]) {
    if parts.len() < local_probs.len() {
        local_probs.truncate(parts.len());
    } else if parts.len() > local_probs.len() {
        local_probs.clear();
        local_probs.extend((0..parts.len()).map(|_| 0.0));
    }
    for (ix, Partition { first, last, .. }) in parts.iter().enumerate() {
        local_probs[ix] = accumulated_range(
            &acc_probs[..],
            Range { start : *first, end : *last + 1 }
        );
    }
    local_probs.truncate(parts.len());
}

fn calculate_local_probs(
    acc_probs : &[f64], 
    local_maxima : &[(Range<usize>, Mode<f64>)], 
    local_probs : &mut Vec<f64>
) {
    local_probs.clear();
    for (ix, (range, _)) in local_maxima.iter().enumerate() {
        local_probs.push(accumulated_range(acc_probs, range.clone()));
    }
    
    // Account for merged intervals
    // local_probs.truncate(local_maxima.len());
}

// Holds a series of merge/sep possibilities.
struct MergeSpace {
    merge_ranges : Vec<Range<usize>>,
    merge_sep : Vec<Range<usize>>,
    local_maxima : Vec<(Range<usize>, Mode<f64>)>
}

impl MergeSpace {

    fn calculate(probs : &[f64], 
        min_mode : usize,
        max_mode : usize,
        min_range : usize,
        max_range : usize
    ) -> Self {
        // Holds as many local maxima as are possible by array size / min_range.
        let max_intervals = probs.len() / min_range;

        // TODO iterate backward from max range up to first divisor
        // of max_merge_sz
        let mut max_merge_sz = 1;
        while max_merge_sz*min_range <= max_range {
            max_merge_sz += 1;
        }

        let mut local_maxima = Vec::new();
        // println!("max intervals = {}", max_intervals);
        for _ in 0..max_intervals {
            // let past_sz = local_maxima.len();
            super::single_interval_search(
                probs,
                &mut local_maxima,
                min_range,
                local_mode,
                false
            );
            // if local_maxima.len() == past_sz {
            //    break;
            // }
        }
        
        // Extend first and last modes (note: this can extend the min_range, max_range) informed by the user.
        if let Some(mut m) = local_maxima.first_mut() {
            m.0.start = 0;
        }
        if let Some(mut m) = local_maxima.last_mut() {
            m.0.end = probs.len();
        }
        
        // Make intervals contiguous.
        let mut n_intervals = local_maxima.len();
        for i in 1..n_intervals {
            let half_dist = (local_maxima[i].0.start - local_maxima[i-1].0.end) / 2;
            if half_dist > 1 {
                local_maxima[i-1].0.end += half_dist;
                local_maxima[i].0.start = local_maxima[i-1].0.end;
            }
        }
        println!("after contiguous = {:?}", local_maxima);
        
        // This step limits the merge graph size to 2^16 = 65_536 nodes.
        'outer : while local_maxima.len() > 16 {
            for i in (0..(local_maxima.len()-1)).rev() {
                merge_intervals(&mut local_maxima, i);
                if local_maxima.len() <= 16 {
                    break 'outer;
                }
            }
        }
        n_intervals = local_maxima.len();
        println!("after merrge = {:?}", local_maxima);

        /*let (valid_merges, merge_ranges) = determine_valid_merges(
            n_intervals,
            min_mode,
            max_mode,
            max_merge_sz
        );
        println!("valid merges={:?}\nmerge ranges={:?}", valid_merges, merge_ranges);
        */

        // Now, visit histogram to find the weighted cross-entropy that most closely matches
        // the marginal cross-entropy.
        let mut merge_ranges = Vec::new();
        let mut merge_sep = Vec::new();
        // let mut state = MergeState { n_contiguous : 1, n_merges : 0 };
        // println!("Mode range = {}-{}", min_mode, max_mode);
        for n_modes in min_mode..(max_mode+1) {
            // println!("For {} modes", n_modes);
            next_merge_range(
                &mut merge_ranges, 
                &mut merge_sep, 
                &mut vec![Range { start : 0, end : 1 }], 
                false, 
                n_modes, 
                n_intervals,
                max_merge_sz
            );
            // state = MergeState { n_contiguous : 1, n_merges : 0 };
            // curr_merge.clear();
            // println!("First branch done");
            next_merge_range(
                &mut merge_ranges, 
                &mut merge_sep, 
                &mut vec![Range { start : 0, end : 1 }], 
                true,
                n_modes,
                n_intervals,
                max_merge_sz
            );
        }
        
        Self { merge_ranges, merge_sep, local_maxima }
    }
    
    
}

/* Gets the partition such that the average distance between mode and median
within each partition is minimized. */
pub fn closest_median_mode_partition(
    probs : &[f64],
    min_mode : usize,
    max_mode : usize,
    min_range : usize,
    max_range : usize,
    min_prob : f64,
    max_med_mode_dist : usize
) -> Vec<Partition> {
    let MergeSpace { 
        merge_ranges, 
        merge_sep, 
        local_maxima 
    } = MergeSpace::calculate(probs, min_mode, max_mode, min_range, max_range);
    println!("{:?}", (min_mode, max_mode, min_range, max_range));
    let acc_probs : Vec<_> = cumulative_sum(probs.iter().copied())
        .collect();
    let acc_marg_entropy : Vec<_> = cumulative_entropy(probs.iter().copied())
        .collect();
    let acc_probs : Vec<_> = cumulative_sum(probs.iter().copied())
        .collect();
    let mut min_dist : f32 = f32::MAX;
    let mut best_part = Vec::new();
    println!("{:?}", local_maxima);
    
    'outer : for merge_s in merge_sep {
        let mut curr_dist = 0.0;
        let mut n_counted = 0;
        'inner : for r in &merge_ranges[merge_s.clone()] {
            let section = Range { 
                start : local_maxima[r.start].0.start, 
                end : local_maxima[r.end-1].0.end
            };
            let local_p = accumulated_range(&acc_probs, section.clone())
                .max(std::f64::EPSILON);
            if local_p < min_prob {
                continue;
            }
            n_counted += 1;
            let mode_ix = local_maxima[r.clone()].iter()
                .max_by(|a, b| a.1.val.partial_cmp(&b.1.val).unwrap_or(Ordering::Equal) )
                .unwrap()
                .1.pos;
            
            let local_probs = &acc_probs[section.clone()];
            let mut median_ix = local_probs
                .partition_point(|p| *p / local_p <= 0.5).min(local_probs.len()-1);
            median_ix += section.start;
            
            let diff = mode_ix.abs_diff(median_ix);
            if diff > max_med_mode_dist {
                continue 'outer;
            }
            
            curr_dist += diff as f32;
        }
        curr_dist /= (n_counted as f32);    
        if n_counted >= min_mode && curr_dist < min_dist {
            min_dist = curr_dist;
            best_part.clear();
            for r in  &merge_ranges[merge_s.clone()] {
                let part = part_from_range(&local_maxima[r.clone()]);
                best_part.push(part);
            }
        }
    }
    best_part.retain(|part| {
        let local_p = accumulated_range(
            &acc_probs, 
            Range { start : part.first, end : part.last + 1 }
        );
        local_p >= min_prob
    });
    best_part
}

/// Finds a histogram partition over an unknown number of modes (assumed to be
/// within a user-provided interval) that minimizes the difference between the
/// marginal histogram entropy and the conditional local histograms cross-entropies
/// wrt. the marginal histogram. If the histogram were to be a mixture distribution,
/// with a perfect analytical description like m(x) = \sum_k p_k(x)theta_k,
/// then weighted conditional entropies should be equal to a simple function of
/// the partial marginal entropies (over the region of where the support of the
/// conditional distribution is nonzero).
/// Since this most likely won't be the case if we define our functions as a partition
/// over the variable domain, we settle for the partition that minimizes the error
/// to this situation.
/// Arguments
/// probs : A probability histogram.
/// min_mode, max_mode: Mode search space
/// min_range, max_range: minimum and maximum size of conditional intervals
pub fn min_partial_entropy_partition(
    probs : &[f64],
    min_mode : usize,
    max_mode : usize,
    min_range : usize,
    max_range : usize
) -> Vec<Partition> {
    let MergeSpace { 
        merge_ranges, 
        merge_sep, 
        local_maxima 
    } = MergeSpace::calculate(probs, min_mode, max_mode, min_range, max_range);
    
    let acc_probs : Vec<_> = cumulative_sum(probs.iter().copied())
        .collect();
    let acc_marg_entropy : Vec<_> = cumulative_entropy(probs.iter().copied())
        .collect();

    // Perhaps we can normalize values by this marginal entropy. Since we know 
    // the summed conditional entropy cannot be lower than the marginal entropy,
    // this ratio gives a value on the 0.0 - 1.0 scale.
    // let marg_entropy = acc_marg_entropy.last().unwrap();
    
    let mut smallest_entropy = f64::MAX;
    let mut best_partition = Vec::new();
    
    for merge_s in merge_sep {
        let mut total_cond_entropy = 0.0;
        for r in &merge_ranges[merge_s.clone()] {
            let section = Range { 
                start : local_maxima[r.start].0.start, 
                end : local_maxima[r.end-1].0.end
            };
            let p = accumulated_range(&acc_probs, section.clone()).max(std::f64::EPSILON);
            assert!(p >= std::f64::EPSILON && p <= 1.0);

            // Gets section of marginal entropy corresponding to this conditional
            let partial_entropy = accumulated_range(&acc_marg_entropy, section.clone());
            assert!(partial_entropy >= 0.0);

            // The actual conditional entropy is a simple function of
            // the partial marginal entropy, derived from a mixture model
            // assumed to contain conditional distributions over the partitions.
            // The term -p.ln() makes the conditional for n_modes == 1 = 0, equal
            // to the marginal.
            let cond_entropy = partial_entropy / p - p.ln();
            assert!(cond_entropy >= 0.0);

            total_cond_entropy += cond_entropy;

            // if total_cond_entropy > smallest_entropy {
            //    break;
            // }
        }

        // Weight total conditional entropy by probability (averaging). Or else sums
        // for different modes won't be comparable (we want cond entropy per #modes),
        // and entropies with higher number of modes would always be preferable.
        let avg_cond_entropy = total_cond_entropy / merge_s.len() as f64;
        
        if avg_cond_entropy < smallest_entropy {
            smallest_entropy = avg_cond_entropy;
            best_partition.clear();
            for r in &merge_ranges[merge_s.clone()] {
                // Get largest mode in this interval
                let part = part_from_range(&local_maxima[r.clone()]);
                best_partition.push(part);
            }
        } else {

        }
    }

    best_partition
}

fn part_from_range(local_maxima : &[(Range<usize>, Mode<f64>)]) -> Partition {
    let mode = local_maxima.iter()
    .max_by(|a, b|
        a.1.val.partial_cmp(&b.1.val).unwrap_or(Ordering::Equal)
    ).unwrap().1.pos;
    let (fst_range, lst_range) = (
        local_maxima.first().cloned().unwrap().0,
        local_maxima.last().cloned().unwrap().0
    );
    Partition {
        mode,
        first : fst_range.start,
        last : lst_range.end - 1
    }
}

// TODO store curr_merge at an allocation arena (the size of allocated
// vec current state is upper bounded by the call graph size). Alternatively,
// do not clear the vector when it fails or ends a path, but rather truncate it
// to the size before the first recursive call was made.
pub fn next_merge_range(
    merge_ranges : &mut Vec<Range<usize>>,
    merge_sep : &mut Vec<Range<usize>>,
    curr_merge : &mut Vec<Range<usize>>,
    should_merge : bool,
    n_modes : usize,
    n_intervals : usize,
    max_merge_sz : usize
) {    
    if should_merge {
        curr_merge.last_mut().unwrap().end += 1;
    } else {
        let last = curr_merge.last().cloned().unwrap();
        curr_merge.push(Range { start : last.end, end : last.end + 1 });
    }
    
    let last = curr_merge.last().cloned().unwrap();
    if last.end - last.start <= max_merge_sz {
        if curr_merge.len() <= n_modes && last.end < n_intervals {
            next_merge_range(
                merge_ranges,
                merge_sep,
                &mut curr_merge.clone(),
                false,
                n_modes,
                n_intervals,
                max_merge_sz
            );
            // TODO truncate vec here to the size before the first recursive call to 
            // avoid the clone.
            next_merge_range(
                merge_ranges,
                merge_sep,
                curr_merge,
                true,
                n_modes,
                n_intervals,
                max_merge_sz
            );
        } else if curr_merge.len() == n_modes && last.end == n_intervals {
            let len_before = merge_ranges.len();
            merge_ranges.extend(curr_merge.drain(..));
            merge_sep.push(Range { start : len_before, end : merge_ranges.len() });
        } else {
            curr_merge.clear();
            return;
        }
    } else {
        curr_merge.clear();
        return;
    }
}

fn local_mode(
    probs : &[f64], 
    range : Range<usize>, 
    interval_sz : usize
) -> Option<(Range<usize>, Mode<f64>)> {    
    if range.end - range.start < interval_sz {
        return None;
    }
    let local_mode = crate::calc::count::mode(&probs[range.clone()]);
    let pos = local_mode.pos + range.start;
    let mut start = pos.saturating_sub(interval_sz / 2).max(range.start);
    let mut end = (pos + interval_sz / 2).min(range.end);
    let mut sz = end - start;
    let mut extended_left = false;
    let mut extended_right = false;
    loop {
        if end - start < interval_sz && start > range.start {
            start -= 1;
            extended_left = true;
        } else {
            extended_left = false;
        }
        if end - start < interval_sz && end < range.end {
            end += 1;
            extended_left = true;
        } else {
            extended_left = false;
        }
        if end - start == interval_sz {
            break;
        } else {
            if !extended_left && !extended_right {
                return None;
            }
        }
    }
    let effective_range = Range { start, end };
    Some((effective_range, Mode { pos, val : local_mode.val }))
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
    let pos = h.partition_point(|a| a.as_() / max_val <= q ).min(h.len()-1);
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
            let ca = accumulated_range(&cumul_probs, (*a).clone());
            let cb = accumulated_range(&cumul_probs, (*b).clone());
            (ca).partial_cmp(&cb).unwrap_or(Ordering::Equal)
        }).unwrap();
        normal_is_likely = (cumul_probs[rare_range.end]-cumul_probs[rare_range.start]) >= min_prob;
    }
    normals
}

/// Represents the empirical cumulative distribution function (ECDF) of a sample.
/// TODO separate AbsCumulative and RelCumulative
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

/* A histogram binning strategy */
// https://en.wikipedia.org/wiki/Histogram
pub enum Binning {

    // Bins between (min, max) at sample with equal intervals of size Width.
    Width(f64),

    // Binds between (min, max) with equal intervals in such a way that there
    // are count intervals.
    Count(u32),

    // Produce ceil(sqrt(n)) bins where n is the sample size.
    CountSqrt,

    // Produces ceil(log2(n)) + 1 bins, where n is the sample size.
    CountSturges,

    // Produces 2*cbrt(n) bins
    CountRice,

    // CountDoane
    // CountScott
    // CountFreedman
    // CountShimazaki

}

#[derive(Debug, Clone)]
pub struct IntervalPartition {

    min : f64,

    max : f64,

    // Size of fragmented intervals
    intv : f64,

    // Full interval to be represented
    ampl : f64,

    n_bins : usize,

    n : u32

}

impl IntervalPartition {

    pub fn new_empty(min : f64, max : f64, binning : Binning) -> Option<Self> {
        assert!(max > min);
        let ampl = max - min;
        let n_bins = match binning {
            Binning::Count(n) => n as usize,
            Binning::Width(w) => (ampl / w).ceil() as usize,
            _ => return None
        };
        let intv = ampl / n_bins as f64;
        Some(Self {
            n : 0,
            min,
            max,
            intv,
            ampl,
            n_bins
        })
    }

    pub fn new<'a>(sample : impl Iterator<Item=&'a f64> + Clone, n_bins : usize) -> Self {
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
        Self { min, max, intv, ampl, n_bins, n }
    }

    pub fn bin_containing(&self, val : f64) -> Option<usize> {
        let bin = ((val - self.min) / self.intv).floor() as usize;
        if bin < self.n_bins {
            Some(bin)
        } else {
            None
        }
    }

    pub fn bin_bounds(&self, pos : usize) -> Option<(f64, f64)> {
        let low = self.min + self.intv*(pos as f64);
        let high = low + self.intv;
        Some((low, high))
    }

}

/// Structure to represent one-dimensional empirical distributions non-parametrically (Work in progress).
/// One-dimensional histogram, useful for representing univariate marginal distributions
/// of sampled posteriors non-parametrically. Retrieved by indexing a Sample structure.
/// The histogram resolution is a linear function of the sample size. Histograms can be
/// thought of as a non-parametric counterpart to bayes::distr::Normal in the sense that it
/// represents a distribution over an unbounded real variable. But unlike Normals, which are
/// defined only by mean and variance, histograms can represent any empiric univariate distribution.
/// Histogram implements Distribution API.
pub trait Histogram {

    fn iter_bins(&self) -> Vec<Bin> {
        (0..self.num_bins()).map(|pos : usize| self.bin(pos).unwrap() ).collect()
    }

    fn bin(&self, pos : usize) -> Option<Bin> {
        let (low, high) = self.bin_bounds(pos)?;
        let prop = self.proportion(pos);
        let count = self.count(pos);
        Some(Bin{ low, high, prop, count })
    }

    fn interval(&self) -> f64;

    fn num_bins(&self) -> usize;

    fn limits(&self) -> (f64, f64);

    /// Returns 0 if bin position is outside bounds or no elements are allocated to it.
    fn count(&self, pos : usize) -> u64;

    /// Returns 0 if bin position is outside bounds or no elements are allocated to it.
    fn proportion(&self, pos : usize) -> f64;

    fn bin_bounds(&self, pos : usize) -> Option<(f64, f64)>;

    fn bin_containing(&self, val : f64) -> Option<usize>;

}

/* A histogram for real-time scenarios. Values are known to
be within the given bounds specified at creation, and counts
can be updated in online mode. Stores counts in a vector, so
the used memory is proportional to the sample interval bounds.
Access to frequencies are done in constant time, so this is
the most performant approach. */
pub struct DenseCountHistogram {
    part : IntervalPartition,

    // Stores intervals of same size, with the interval order as key and count of ocurrences
    // as values. Intervals without an element count are missing entries.
    bins : Vec<u64>
}

impl DenseCountHistogram {

    pub fn increment(&mut self, pos : usize, by : u64) {
        self.bins[pos] += by;
        self.part.n += by as u32;
    }

    pub fn increment_containing(&mut self, val : f64, by : u64) {
        let pos = self.bin_containing(val).unwrap();
        self.increment(pos, by);
    }

    pub fn calculate<'a>(sample : impl Iterator<Item=&'a f64> + Clone, n_bins : usize) -> Self {
        let part = IntervalPartition::new(sample.clone(), n_bins);
        let mut bins : Vec<_> = (0..part.n_bins).map(|_| 0 ).collect();
        for s in sample {
            let b = part.bin_containing(*s).unwrap();
            bins[b] += 1;
        }
        Self { part, bins }
    }

}

impl Histogram for DenseCountHistogram {

    fn num_bins(&self) -> usize {
        self.part.n_bins
    }

    fn interval(&self) -> f64 {
        self.part.intv
    }

    fn limits(&self) -> (f64, f64) {
        (self.part.min, self.part.max)
    }

    /// Returns 0 if bin position is outside bounds or no elements are allocated to it.
    fn count(&self, pos : usize) -> u64 {
        self.bins.get(pos).cloned().unwrap_or(0u64)
    }

    /// Returns 0 if bin position is outside bounds or no elements are allocated to it.
    fn proportion(&self, pos : usize) -> f64 {
        self.bins.get(pos).map(|b| *b as f64 / self.part.n as f64 ).unwrap_or(0.0)
    }

    fn bin_bounds(&self, pos : usize) -> Option<(f64, f64)> {
        self.part.bin_bounds(pos)
    }

    fn bin_containing(&self, val : f64) -> Option<usize> {
        self.part.bin_containing(val)
    }

}

/* Holds data in a BTreeMap mapping bin order to frequency count.
Is slightly more demanding to compute than DenseCountHistogram,
but saves up memory since bins with zero counts aren't stored. */
#[derive(Debug)]
pub struct SparseCountHistogram {

    part : IntervalPartition,

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

impl Histogram for SparseCountHistogram {

    fn num_bins(&self) -> usize {
        self.part.n_bins
    }

    fn interval(&self) -> f64 {
        self.part.intv
    }

    fn limits(&self) -> (f64, f64) {
        (self.part.min, self.part.max)
    }

    /// Returns 0 if bin position is outside bounds or no elements are allocated to it.
    fn count(&self, pos : usize) -> u64 {
        *self.bins.get(&(pos as u64)).clone().unwrap_or(&0)
    }

    /// Returns 0 if bin position is outside bounds or no elements are allocated to it.
    fn proportion(&self, pos : usize) -> f64 {
        self.bins.get(&(pos as u64)).map(|b| *b as f64 / self.part.n as f64 ).unwrap_or(0.0)
    }

    fn bin_bounds(&self, pos : usize) -> Option<(f64, f64)> {
        self.part.bin_bounds(pos)
    }

    fn bin_containing(&self, val : f64) -> Option<usize> {
        self.part.bin_containing(val)
    }

}

impl SparseCountHistogram {

    pub fn increment(&mut self, pos : u64, by : u64) {
        let b_count = self.bins.entry(pos).or_insert(0);
        *b_count += 1;
        self.part.n += by as u32;
    }

    pub fn increment_containing(&mut self, val : f64, by : u64) {
        let pos = self.bin_containing(val).unwrap();
        self.increment(pos as u64, by);
    }

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
        let part = IntervalPartition::new(sample.clone(), n_bins);
        let mut bins = BTreeMap::<u64, u64>::new();
        for s in sample {

            // Retrieves the discrete bin allocation for s as the ordered interval
            let b = part.bin_containing(*s).unwrap() as u64;
            let b_count = bins.entry(b).or_insert(0);
            *b_count += 1;
        }

        Self { part, bins }
    }

}

/// Pair of marginal histograms.
pub struct MarginalHistogram {
    bottom : DenseCountHistogram,
    right : DenseCountHistogram
}

impl MarginalHistogram {

    pub fn build(a : DenseCountHistogram, b : DenseCountHistogram) -> Self {
        Self{ bottom : a, right : b}
    }

}

/*#[derive(Debug, Clone, Copy)]
pub struct JointMean {
    pub x : f32,
    pub y : f32
}*/

// A dense histogram defined over a 2D surface.
// Useful to represent the joint distribution of pairs of non-parametric
// posterior parameters and to calculate pair-wise statistics for them, such
// as their covariance and correlation. This can be thought of as a nonparametric
// counterpart to bayes::prob::MultiNormal of dimensionality 2. Representing distributions
// of higher-dimensions non-parametrically becomes computationally infeasible for dimensionality
// greater than two, so you must resort to a parametric formulation in this case; or at least
// to a mixture.
/* The most performant implementation JointCumulative would increment all
(i, j) such that i < y, j < x, so that probability queries would be done
in constant time */
/*#[derive(Debug, Clone)]
pub struct JointHistogram {
    vals : DMatrix<u32>,
    px : PartitionInterval,
    py : PartitionInterval,
}

impl<T> JointHistogram<T>
where
    T : AddAssign + Add<Output=T> + Clone + Copy + std::fmt::Debug + PartialEq + Eq + PartialOrd + Ord + num_traits::Zero +
    Div<Output=T> + num_traits::Zero + num_traits::ToPrimitive + std::iter::Sum + 'static
{

    pub fn new(height : usize, width : usize) -> Self {
        Self { vals : (0..(height*width)).map(|_| T::zero() ).collect(), height, width, total : T::zero() }
    }

    pub fn increment(&mut self, x : usize, y : usize, n : T) {
        self.vals[self.width * y + x] += n;
        self.total += n;
    }

    pub fn iter_x_cond_y<'a>(&'a self, y : usize) -> impl Iterator<Item=&'a T> + 'a {
        let start = (self.width*y);
        self.vals[start..(start+self.width)].iter()
    }

    pub fn iter_y_cond_x<'a>(&'a self, x : usize) -> impl Iterator<Item=&'a T> + 'a {
        self.vals[x..].iter().step_by(self.width)
    }

    pub fn marginal_probability_y(&self, y : usize) -> f32 {
        self.iter_x_cond_y(y).copied().sum::<T>().to_f32().unwrap() / self.total.to_f32().unwrap()
    }

    pub fn marginal_probability_x(&self, x : usize) -> f32 {
        self.iter_y_cond_x(x).copied().sum::<T>().to_f32().unwrap() / self.total.to_f32().unwrap()
    }

    pub fn joint_probability(&self, x : usize, y : usize) -> f32 {
        self.vals[y*self.width + x].to_f32().unwrap() / self.total.to_f32().unwrap()
    }

    /*pub fn mean(&self) -> JointMean {
        let (mut mx, mut my) = (0.0, 0.0);
        for y in 0..self.height {
            for x in 0..self.width {
                let p = self.vals[y*self.width + x].to_f32().unwrap() / self.total.to_f32().unwrap();
                mx += p*(x as f32);
                my += p*(y as f32);
            }
        }
        JointMean { x : mx, y : my }
    }*/

}*/

pub struct JointCumulative {
    vals : DMatrix<u32>
}

impl JointCumulative {

    pub fn new(height : usize, width : usize) -> Self {
        Self { vals : DMatrix::<u32>::zeros(height, width) }
    }

    pub fn increment(&mut self, y : usize, x : usize, by : u32) {
        let h = self.vals.nrows();
        let w = self.vals.ncols();
        let w_right = w - x;
        let w_left = w - w_right;
        let h_below = h - y;

        // Increment everything to the right of x
        self.vals.slice_mut((0, x), (w_right, h)).add_scalar_mut(by);

        // Increment everything below and to the left of (y, x)
        self.vals.slice_mut((y, 0), (w_left, x)).add_scalar_mut(by);
    }

}

// https://en.wikipedia.org/wiki/Interpolation
pub trait Interpolator {

}

pub struct NearestInterpolator {
    // Set value = closest value. This generate step-like
    // functions
}

pub struct LinearInterpolator {
    // y == ya + (yb - ya) * ((x - xa) / (xb - xa))
}

// Generalizes the linear interpolator, and builds the full
// polynomial of degree=# points that interpolates the values.
// Might be costly for many points.
pub struct PolynomialInterpolator {

}

// Based on https://elonen.iki.fi/code/tpsdemo/index.html
pub struct SplineInterpolator {
    a : Vector3<f64>,
    domain : DMatrix<f64>,
    w : DVector<f64>,
    metrics : DVector<f64>
}

fn spline_rbf(r : f64) -> f64 {
    if r > 0.0 { r.powf(2.) * r.ln() } else { 0.0 }
}

impl SplineInterpolator {

    // Interpolate (x, y) into f(x, y)
    pub fn interpolate(&mut self, x : f64, y : f64) -> f64 {
        let v = Vector2::new(x, y);
        let l1 = LpNorm(1);
        for i in 0..self.w.len() {
            self.metrics[i] = spline_rbf(self.domain.column(i).apply_metric_distance(&v, &l1));
        }
        Vector3::new(1.0, x, y).dot(&self.a) + self.w.dot(&self.metrics)
    }

    // Lambda 0.0 = perfect interpolation, up to 1.0
    pub fn new(xs : &[f64], ys  : &[f64], zs : &[f64], lambda : f64) -> Option<Self> {
        let n = xs.len();

        let mut domain = DMatrix::zeros(2, n);
        for i in 0..n {
            domain[(0, i)] = xs[i];
            domain[(1, i)] = ys[i];
        }

        let mut pts_mat = DMatrix::from_element(n, 3, 1.0);
        pts_mat.column_mut(1).copy_from_slice(xs);

        // pts_mat.column_mut(2).copy_from_slice(ys);
        for i in 0..n {
            // Unless this noise is added, the matrix L will not be of full rank.
            if xs[i] == ys[i]  {
                pts_mat[(i, 2)] = ys[i] + rand::random::<f64>()*1e-10;
            } else {
                pts_mat[(i, 2)] = ys[i];
            }
        }

        // TODO calculate only upper part
        let mut metric_diffs = DMatrix::zeros(n, n);
        let l1 = LpNorm(1);
        for i in 0..n {
            for j in 0..n {
                metric_diffs[(i,j)] = domain.column(i).apply_metric_distance(&domain.column(j), &l1);
                // metric_diffs[(i,j)] = (domain.column(i).clone_owned() - &domain.column(j)).abs().sum();
            }
        }

        // Mean distances between control points xy's projections
        let alpha = metric_diffs.sum() / (n as f64).powf(2.);

        let mut L = DMatrix::zeros(n + 3, n + 3);
        let I_lambda = DMatrix::<f64>::identity(n, n).scale(lambda);

        // TODO calculate only upper part
        for i in 0..n {
            for j in 0..n {

                // TODO only needs to be added to L(i,j) on diagonal
                let s : f64 = I_lambda[(i,j)]*alpha.powf(2.);

                L[(i,j)] = spline_rbf(metric_diffs[(i,j)]) + s;
            }
        }
        L.slice_mut((0, n), (n, 3)).copy_from(&pts_mat);
        L.slice_mut((n, 0), (3, n)).tr_copy_from(&pts_mat);

        let mut b = DVector::zeros(n+3);
        b.rows_mut(0, n).copy_from_slice(&zs);
        //println!("{:.2}", L);
        //println!("{:.2}", b);
        //println!("Det(L) = {:.4}", L.determinant());
        //println!("rank(L) = {}", L.rank(0.01));
        assert!(L.nrows() == b.nrows());
        let x = LU::new(L).solve(&b)?;
        // println!("Solve ok");
        let nx = x.len();
        Some(Self {
            metrics : DVector::zeros(n),
            w : x.rows(0, n).clone_owned(),
            a : Vector3::new(x[nx-3], x[nx-2], x[nx-1]),
            domain
        })
    }

}
// Uses a weighted combination of function values, with weights
// proportional to the distances to all points in the set (then
// divide result by weight sum). Might work well for closely
// packed points, but the interpolated value decays to zero
// at large intervals and at extrapolation. Alternatively,
// uses only the closest points within a sphere around the
// desired value.
// https://en.wikipedia.org/wiki/Inverse_distance_weighting
pub struct InvDistInterpolator {

}

impl Interpolator for BilinearInterpolator { }

use nalgebra::Matrix4;

// https://en.wikipedia.org/wiki/Bilinear_interpolation
#[derive(Debug, Clone)]
pub struct BilinearInterpolator {
    xs : Vec<f64>,
    ys : Vec<f64>,
    zs : Vec<f64>
}

impl BilinearInterpolator {

    // Important: xs and ys are assumed sorted.
    pub fn new(xs : &[f64], ys : &[f64], zs : &[f64]) -> Self {
        Self {
            xs : xs.to_vec(),
            ys : ys.to_vec(),
            zs : zs.to_vec()
        }
    }

    /*// This is for the regular grid case. To generalize to
    // arbitrarily positioned 4 points (trapezoid), see
    // https://math.stackexchange.com/questions/828392/spatial-interpolation-for-irregular-grid
    pub fn eval(&self, x_val : f64, y_val : f64) -> f64 {

        let ix_x = self.xs.partition_point(|x| x < x_val );
        let ix_y = self.ys.partition_point(|y| y < y_val );

        // In a equally-spaced grid, the partition points xs < x, xs > x
        // and ys < x, ys > x WILL be the four closest points, and they
        // will delimit the shaded region that contains the test point
        // but not any point in the original set. This is not the case
        // for irregularly-sampled data.
        let pts = [
            (self.xs[ix_x], self.ys[ix_x]);
            (self.xs[ix_x+1], self.ys[ix_x+1]);
            (self.xs[ix_y], self.ys[ix_y]);
            (self.xs[ix_y+1], self.ys[ix_y+1])
        ];

        let mut a = Matrix4::ones();
        for i in 0..4 {
            a[(i, 1)] = pts[i].0;
            a[(i, 2)] = pts[i].1;
            a[(i, 3)] = pts[i].0 * pts[i].1;
        }
        let b = DVector::from_vec(zs.to_vec());
        let coefs = a.solve(b);
        coefs * Vector4::new(1.0, x_val, y_val, x_val*y_val)
    }*/

}

// cargo test --lib -- mvinterp --nocapture
#[test]
fn mvinterp() {
    use statrs::distribution::*;
    let n = Normal::new(0.0, 1.0).unwrap();
    let domain : Vec<_> = (0..10).map(|x| -3.0 + (x as f64 / 10.0)*6.0 ).collect();
    let domain_r : Vec<_> = (0..10).map(|x| domain[x] + rand::random::<f64>()*0.0001 ).collect();
    let height : Vec<_> = domain.iter().map(|x| n.pdf(*x) ).collect();

    let mut inter = SplineInterpolator::new(&domain, &domain, &height, 0.1).unwrap();
    let mut out_x = Vec::new();
    let mut out_y = Vec::new();
    let mut out_z = Vec::new();
    for i in 0..100 {
        let x = -3.0 + (i as f64 / 100.0)*6.0;
        out_x.push(x);
        out_y.push(x);
        out_z.push(inter.interpolate(x, x));
    }
    // crate::save!("/home/diego/Downloads/interp.json", out_x, out_y, out_z);
    // let mut inter = polyspline::PolyharmonicSpline::new_surface(2, &domain, &domain_r, &height, None);
}

// Translated from the original Julia code by
// https://gist.github.com/lstagner/04a05b120e0be7de9915
// Licensed under MIT.
mod polyspline {

    use nalgebra::{DVector, DMatrix};
    use std::ops::AddAssign;

    pub struct PolyharmonicSpline {
        dim : usize,
        order : u32,
        coeff : DVector<f64>,
        centers : DMatrix<f64>,
        error : f64
    }

    impl PolyharmonicSpline {

        /* Return function values at the given points. */
        fn interpolate_generic(
            &self,
            x : &DMatrix<f64>
        ) -> DVector<f64> {
            let (m, n) = x.shape();
            assert!(n == self.dim);
            let mut interpolates = DVector::zeros(m);
            for i in 0..m {
                let mut tmp = 0.0;
                let l = self.coeff.nrows() - (n+1);
                for j in 0..l {
                    let norm = x.row(i).metric_distance(&self.centers.row(j));
                    tmp += self.coeff[j]*polyharmonic_k(norm, self.order);
                }

                tmp += self.coeff[l];
                for j in 1..(n+1) {
                    tmp += self.coeff[l+j]*x[(i, j-1)];
                }
                interpolates[i] = tmp;
            }
            interpolates
        }

        // Eval interp points at x
        pub fn interpolate_curve(&self, x : &[f64]) -> DVector<f64> {
            let mut dm = DMatrix::zeros(x.len(), 1);
            dm.column_mut(0).copy_from_slice(x);
            self.interpolate_generic(&dm)
        }

        // Eval interp points at domain (x,y)
        pub fn interpolate_surface(&self, x : &[f64], y : &[f64]) -> DVector<f64> {
            let mut dm = DMatrix::zeros(x.len(), 2);
            dm.column_mut(0).copy_from_slice(x);
            dm.column_mut(1).copy_from_slice(x);
            self.interpolate_generic(&dm)
        }

        pub fn new_curve(
            k : u32,
            xs : &[f64],
            ys : &[f64],
            s : Option<f32>
        ) -> Self {
            Self::new_generic(
                k,
                DMatrix::from_vec(xs.len(), 1, xs.to_vec()),
                DMatrix::from_vec(ys.len(), 1, ys.to_vec()),
                s
            )
        }

        pub fn new_surface(
            k : u32,
            xs : &[f64],
            ys : &[f64],
            zs : &[f64],
            s : Option<f32>
        ) -> Self {
            let mut centers = DMatrix::zeros(xs.len(), 2);
            centers.column_mut(0).copy_from_slice(xs);
            centers.column_mut(1).copy_from_slice(ys);
            Self::new_generic(k, centers, DMatrix::from_vec(zs.len(), 1, zs.to_vec()), s)
        }

        // For 2D domain:
        // Centers is nx2 domain points,
        // values is nx1 (z, fn eval dimension)
        // For 1D domain:
        // Centers is nx1 domain points
        // values is nx1 eval dimension.
        // K is the RBF order >= 1.
        pub fn new_generic(
            k : u32,
            centers : DMatrix<f64>,
            values : DMatrix<f64>,
            s : Option<f32>
        ) -> Self {
            let s = s.unwrap_or(0.0);
            let (m, n) = centers.shape();
            assert!(m == values.nrows());
            let mut M = DMatrix::zeros(m, m);
            let mut N = DMatrix::zeros(m, n+1);

            for i in 0..m {
                let nc = N.ncols();
                N[(i,0)] = 1.0;
                N.row_mut(i).columns_mut(1, nc-1).copy_from(&centers.row(i));
                for j in 0..m {
                    let norm = centers.row(i).metric_distance(&centers.row(j));
                    M[(i,j)] = polyharmonic_k(norm, k);
                }
            }

            let S = DMatrix::identity(m, m).scale(s as f64);
            M.add_assign(&S);

            let Nt = N.transpose();

            let mut L = DMatrix::zeros(M.nrows() + Nt.nrows(), M.ncols() + N.ncols());
            L.slice_mut((0,0), M.shape()).copy_from(&M);
            L.slice_mut((0, M.ncols()), N.shape()).copy_from(&N);
            L.slice_mut((M.nrows(), 0), Nt.shape()).copy_from(&Nt);

            // TODO stuck here..
            println!("Calculating pinv");
            let L_pinv = L.pseudo_inverse(0.1).unwrap();
            println!("Pinv calculated");

            let mut y = DVector::zeros(values.nrows() + n + 1);
            y.rows_mut(0, values.nrows()).copy_from(&values);
            let w = L_pinv * y;

            let mut ivalues = DVector::zeros(m);
            for i in 0..m {
                let mut tmp = 0.0;
                for j in 0..m {
                    let norm = centers.row(i).metric_distance(&centers.row(j));
                    tmp += w[j] * polyharmonic_k(norm, k);
                }
                tmp += w[m+1];
                for j in 1..(n+1) {
                    tmp += w[m+j]*centers[(i, j-1)];
                }
                ivalues[i] += tmp;
            }
            let error = values.metric_distance(&ivalues);
            Self { dim : n, order : k, coeff : w, centers, error }
        }

    }

    /* The RBF phi(r) = r^k for k odd; r^k*ln(r) for k even.
    where r is the euclidian metric between data point x
    and center c. */
    fn polyharmonic_k(r : f64, k : u32) -> f64 {
        assert!(k >= 1);
        if k % 2 == 0 {
            r.powf(k as f64) * r.ln()
        } else {
            r.powf(k as f64)
        }
    }

}

