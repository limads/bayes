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

/* If two values are equal, return 1.0; else return 0.0. Useful to map
realizations of categorical in the integers [0..k] to a real weight.
This is the indicator function, a.k.a. Iverson Bracket, for the
logical predicate being the argument a belongs to the set with the
unique element B. */
pub fn indicator<T : Eq>(a : T, b : T) -> f64 {
    if a == b { 1.0 } else { 0.0 }
}

// TODO possibly separate into Binary (logit/sigmoid conversions) / Continuous (standardization conversions)/
// Count (combinatorial calculations). Also, create calc::special to hold the gamma and beta functions.

/// Functions applicable to distribution realizations and parameters,
/// possibly executed by applying vectorized instructions.
pub trait Variate {

    // Convertes a probability in [0-1] to an odds on [-inf, inf], which is a non-linear function.
    fn odds(&self) -> Self;

    // natural logarithm of the odds. Unlike the odds, the logit is a linear function.
    fn logit(&self) -> Self;

    // Converts a logit on [-inf, inf] back to a probability on [0,1]
    fn sigmoid(&self) -> Self;

    fn center(&self, mean : &Self) -> Self;

    fn unscale(&self, inv_scale : &Self) -> Self;

    fn standardize(&self, mean : &Self, inv_scale : &Self) -> Self;

    fn identity(&self) -> Self;

}

impl Variate for f64 {

    fn identity(&self) -> Self {
        *self
    }

    fn odds(&self) -> f64 {
        *self / (1. - *self)
    }

    // The logit or log-odds of a probability p \in [0,1]
    // is the natural log of the ratio p/(1-p)
    fn logit(&self) -> Self {
        let p = if *self == 0.0 {
            f64::EPSILON
        } else {
            *self
        };
        (p / (1. - p)).ln()
    }

    // The sigmoid is the inverse of the logit (or log-odds function).
    // and is given by 1 / (1 + exp(-logit))
    fn sigmoid(&self) -> Self {
        1. / (1. + (-1. * (*self)).exp() )
    }

    fn center(&self, mean : &Self) -> Self {
        *self - *mean
    }

    ///
    /// "Unscales" a value by multiplying it by its precision.
    ///
    /// ```rust
    /// bayes::calc::unscale(1.0, 2.0);
    /// ```
    ///
    /// ```lua
    /// bayes.calc.unscale(1, 0.2)
    /// ```
    fn unscale(&self, inv_scale : &Self) -> Self {
        *self / *inv_scale
    }

    fn standardize(&self, mean : &Self, inv_scale : &Self) -> Self {
        self.center(mean).unscale(inv_scale)
    }

}

/* Normalizing wrt maximum is useful if values are odds-ratios. Then the
ratio to the maximum odds rato gives a probability. */
pub fn odds_to_probs(bins : &[u32]) -> Vec<f32> {
    let max = bins.iter().copied().max().unwrap() as f32;
    bins.iter().map(move |count| *count as f32 / max ).collect()
}

pub fn counts_to_probs_static<const N : usize>(bins : &[u32; N]) -> [f32; N] {
    let max = bins.iter().sum::<u32>() as f32;
    bins.map(move |count| count as f32 / max )
}

/* Normalizing wrt sum of values is useful if values are absolute
frequencies. The ratio to the sum then gives a probability */
pub fn counts_to_probs(bins : &[u32]) -> Vec<f32> {
    let max = bins.iter().sum::<u32>() as f32;
    bins.iter().map(move |count| *count as f32 / max ).collect()
}

// For each probability entry in the iterator, calculate the entropy
// for all probs up to the desired point.
pub fn cumulative_entropies<'a>(probs : impl Iterator<Item=f32> + 'a) -> impl Iterator<Item=f32> + 'a {
    running::cumulative_sum(probs.map(move |p| p * p.ln() ))
}

// Calculate the total entropy for an iterator over probabilities.
pub fn entropy(probs : impl Iterator<Item=f32>) -> f32 {
    (-1.)*probs.fold(0.0, |s, p| s + p * p.ln() )
}

pub fn squared_deviations<'a>(vals : impl Iterator<Item=f32> + 'a, m : f32) -> impl Iterator<Item=f32>  + 'a {
    vals.map(move |v| (v - m).powf(2.) )
}

pub fn cumulative_squares<'a>(vals : impl Iterator<Item=f32> + 'a) -> impl Iterator<Item=f32>  + 'a {
    running::cumulative_sum(vals.map(move |v| v.powf(2.) ) )
}

pub fn cumulative_squared_deviations<'a>(vals : impl Iterator<Item=f32> + 'a, m : f32) -> impl Iterator<Item=f32> + 'a {
    running::cumulative_sum(vals.map(move |v| (v - m).powf(2.) ))
}

/// Single-pass univariate statistical calculations.
pub mod running {

    use super::*;
    use std::borrow::Borrow;
    use num_traits::Zero;
    use std::ops::*;

    pub fn cumulative_sum<T>(iter : impl Iterator<Item=T>) -> impl Iterator<Item=T>
    where
        T : AddAssign + Zero + Copy
    {
        iter.scan(
            T::zero(),
            |state : &mut T, it : T| {
                *state += it;
                Some(*state)
            }
        )
    }

    pub fn single_pass_sum_sum_sq(sample : impl IntoIterator<Item=impl Borrow<f64>>) -> (f64, f64, usize) {
        let mut n = 0;
        let (sum, sum_sq) = sample.into_iter().fold((0.0, 0.0), |accum, d| {
            n += 1;
            let (sum, sum_sq) = (accum.0 + *d.borrow(), accum.1 + d.borrow().powf(2.));
            (sum, sum_sq)
        });
        // let n = data.len() as f64;
        let mean = sum / (n as f64);
        (mean, sum_sq / (n as f64) - mean.powf(2.), n)
    }

    pub fn mean(d : impl Iterator<Item=f64>, count : usize) -> f64 {
        d.sum::<f64>() / count as f64
    }

    pub fn mean_variance(d : impl Iterator<Item=f64>, count : usize, unbiased : bool) -> (f64, f64) {
        let (sum, sum_sq) = d.map(|s| (s, s.powf(2.)) )
            .fold((0.0, 0.0), |acc, s| (acc.0 + s.0, acc.1 + s.1) );
        let n = count as f64;
        let mean = sum / n;
        let var = (sum_sq - sum.powf(2.) / n) / if unbiased { n - 1. } else { n };
        (mean, var)
    }

    pub fn mean_variance_from_slice(d : &[f64], unbiased : bool) -> (f64, f64) {
        mean_variance(d.iter().cloned(), d.len(), unbiased)
    }

    /*pub fn mean_absdev(d : &[f64], unbiased : bool) -> (f64, f64) {
        let n = d.len() as f64;
        let (sum, sum_sq) = d.iter()
            .map(|s| (*s, s.powf(2.)) )
            .fold((0.0, 0.0), |acc, s| (acc.0 + s.0, acc.1 + s.1) );
        let mean = sum / n;
        let var = (sum_sq - sum.powf(2.) / n) / if unbiased { n - 1. } else { n };
        (mean, var)
    }*/

    // TODO median_absdev, mean_absdev, mean_min_max, median_min_max, min_max

    /// Assuming s is an iterator over a count of some quantity, iterate over it util the given ratio
    /// of this quantity is reached. When it is reached, return the position at the iterator
    /// where this quantity was reached. Optionally, start with a baseline position and count (e.g. calculate an
    /// upper quantile from a previous iteration accumulated value). The total number of elements should
    /// be known beforehand and passed as the second argument.
    pub fn next_quantile<S>(
        s : &mut impl Iterator<Item=S>,
        baseline : Option<(usize, S)>,
        total : S,
        quantile : f64
    ) -> Option<(usize, S)>
    where
        S : Add<Output=S> + Zero + Copy + Unsigned + AddAssign + ToPrimitive + Ord,
        f64 : NumCast
    {
        assert!(quantile >= 0.0 && quantile <= 1.0);
        let total : f64 = NumCast::from(total).unwrap();
        let mut curr_count = baseline.map(|b| b.1 ).unwrap_or(S::zero());
        let mut curr_ratio = <f64 as NumCast>::from(curr_count).unwrap() / total;
        let mut bin : usize = 0;
        while let Some(s) = s.next() {
            curr_count += s;
            curr_ratio = <f64 as NumCast>::from(curr_count).unwrap() / total;
            if curr_ratio >= quantile {
                let bl = baseline.map(|b| b.0 ).unwrap_or(0);
                assert!(curr_ratio <= 1.0 && <f64 as NumCast>::from(curr_count).unwrap() <= total);
                return Some((bin as usize + bl, curr_count));
            }
            bin += 1;
        }
        None
    }

    /// Given a sequence of counts and a set of quatiles in [0,1], return how many elements
    /// are within each quantile.
    pub fn quantiles<S>(
        mut s : impl Iterator<Item=S>,
        total : S,
        quantiles : &[f64]
    ) -> Vec<(usize, S)>
    where
        S : Add<Output=S> + Zero + Copy + Unsigned + AddAssign + ToPrimitive + Ord,
        f64 : NumCast
    {
        let mut qs = Vec::new();
        for q in quantiles.iter() {
            if let Some(new_q) = next_quantile(&mut s, qs.last().cloned(), total, *q) {
                qs.push(new_q);
            } else {
                return qs;
            }
        }
        qs
    }

}

pub struct Ranks<T> {
    pub min : T,
    pub max : T,
    pub q25 : T,
    pub q75 : T,
    pub median : T
}

pub fn baseline_median<T>(s : &mut [T]) -> T
where
    T : Copy + PartialOrd
{
    baseline_quantile(s, 0.5)
}

pub fn baseline_quantile<T>(s : &mut [T], rank : f64) -> T
where
    T : Copy + PartialOrd
{
    assert!(rank >= 0.0 && rank <= 1.0 );
    s.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal) );
    s[(s.len() as f64 * rank).floor() as usize]
}

impl<T> Ranks<T>
where
    T : Copy + PartialOrd + Sub<Output=T>
{

    pub fn iqr(&self) -> T {
        self.q75 - self.q25
    }

    pub fn calculate(vals : &[T]) -> Self {
        let mut v : Vec<_> = vals.iter().copied().collect();
        v.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal) );
        let n = v.len();
        let nf = n as f64;
        Ranks {
            min : v[0],
            max : v[n-1],
            q25 : v[(nf * 0.25) as usize],
            q75 : v[(nf * 0.75) as usize],
            median : v[n/2]
        }
    }

}

/*// Based on: https://brilliant.org/wiki/median-finding-algorithm/
// Elements in the list must be distinct
fn recursive_rank_find(mut lst : &mut [T], rank  : usize)
where
    T : PartialOrd
{

    let mut sublists = Vec::new();
    for j in 0..a.len().step_by(5) {
        let sub = &mut lst[j..(j+5)];
        sub.sort_by(|a, b| a.partital_cmp(&b).unwrap_or(Ordering::Equal) );
        sublists.push(sub);
    }

    let mut medians = Vec::new();
    for sub in sublists {
        medians.push(sub[sub.len() / 2]);
    }

    let mut pivot = medians[0];
    if medians.len() <= 5 {
        medians.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal) );
        pivot = medians[medians.len()/2];
    } else {
        pivot = recursive_rank_find(medians, len(medians)/2);
    }

    // Just swap elements in A inplace: if i < pivot and list[i] < pivot, do nothing. Else
    // swap by an element to the right of pivot. This is done in linear time. The elements
    // will still be unordered in the half sub-lists, but they will be smaller than or greater
    // than the pivot.
    let mut low : Vec<_> = lst.iter().filter(|s| *s < pivot ).copied().collect();
    let mut high : Vec<_> = lst.iter().filter(|s| *s > pivot ).copied().collect();
    let k = low.len();
    if rank < k {
        recursive_rank_find(&mut low[..], rank)
    } else if rank > k {
        recursive_rank_find(&mut high[..], rank-k-1)
    } else {
        pivot
    }
}*/

#[test]
fn quantile() {
    let mut s = (0..100).map(|s| 1u64 );
    let s_sum = s.clone().sum::<u64>();
    /*let q_25 = running::quantile(&mut s, None, s_sum, 0.25).unwrap();
    let q_75 = running::quantile(&mut s, Some(q_25), s_sum, 0.75).unwrap();
    println!("Q25 = {:?}; Q75 = {:?}", q_25, q_75);
    println!("Q25/Q75 = {:?};", running::quantiles((0..100).map(|s| 1u64 ), s_sum, &[0.25, 0.75]));*/

}

// TODO calculate mean with slice::select_nth_unstable_by
// or select_nth_unstable.

// Based on https://rosettacode.org/wiki/Cumulative_standard_deviation
pub struct CumulativeStandardDeviation {
    n: f64,
    sum: f64,
    sum_sq: f64
}

impl CumulativeStandardDeviation {
    pub fn new() -> Self {
        CumulativeStandardDeviation {
            n: 0.,
            sum: 0.,
            sum_sq: 0.
        }
    }

    fn push(&mut self, x: f64) -> f64 {
        self.n += 1.;
        self.sum += x;
        self.sum_sq += x * x;

        (self.sum_sq / self.n - self.sum * self.sum / self.n / self.n).sqrt()
    }
}