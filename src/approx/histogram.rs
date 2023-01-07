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
    println!("{:?}", (&best_part, &min_dist));
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
            // println!("end: {}; start: {}; diff: {}", acc_marg_entropy[section.end-1], acc_marg_entropy[section.start], partial_entropy);
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
        
            // println!("Partition {:?} has smaller entropy {} (prev = {:?} with {})", &merge_ranges[merge_s.clone()], total_cond_entropy, best_partition, smallest_entropy);
            
            smallest_entropy = avg_cond_entropy;
            best_partition.clear();
            for r in &merge_ranges[merge_s.clone()] {
                // Get largest mode in this interval
                let part = part_from_range(&local_maxima[r.clone()]);
                best_partition.push(part);
            }
        } else {
            // println!("Partition {:?} has entropy {}", &merge_ranges[merge_s.clone()], total_cond_entropy);
        }
    }
    // println!("{:?}", best_partition);
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
    
    // println!("Considering: {:?} ({})", curr_merge, n_modes);
    
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
            // println!("First branch done");
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
            //println!("n modes = {}; n intervals = {}, max_merge_sz = {}, merges = {:?}",
            //    n_modes, n_intervals, max_merge_sz, curr_merge);
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
    // println!("{:?}", range);
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

    pub fn interval(&self) -> f64 {
        self.intv
    }

    pub fn limits(&self) -> (f64, f64) {
        (self.min, self.max)
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

/*/* The most performant implementation JointCumulative would increment all
(i, j) such that i < y, j < x, so that probability queries would be done
in constant time */
#[derive(Debug, Clone)]
pub struct JointEmpirical<T>
where
    T : AddAssign + Add<Output=T> + Clone + Copy + std::fmt::Debug + PartialEq + Eq + PartialOrd + Ord + num_traits::Zero +
    Div<Output=T> + num_traits::Zero + num_traits::ToPrimitive + std::iter::Sum + 'static
{
    vals : Vec<T>,
    width : usize,
    height : usize,
    total : T
}

#[derive(Debug, Clone, Copy)]
pub struct JointMean {
    pub x : f32,
    pub y : f32
}

impl<T> JointEmpirical<T>
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

    pub fn mean(&self) -> JointMean {
        let (mut mx, mut my) = (0.0, 0.0);
        for y in 0..self.height {
            for x in 0..self.width {
                let p = self.vals[y*self.width + x].to_f32().unwrap() / self.total.to_f32().unwrap();
                mx += p*(x as f32);
                my += p*(y as f32);
            }
        }
        JointMean { x : mx, y : my }
    }

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


