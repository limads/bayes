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
use std::ops::Mul;
use num_traits::AsPrimitive;

/* The entropy of a bimodal histogram with unequal probabilities equals
the entropy of the corresponding unimodal histogram with same mass (due to additivity of entropy).
An algorithm to search for histogram partition points is then to find points such that the local
entropies are much lower than the global entropy. */

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

    fn standardize(&self, mean : &Self, stddev : &Self) -> Self;

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

    fn standardize(&self, mean : &Self, stddev : &Self) -> Self {
        (*self - *mean) / *stddev
    }

}

impl Variate for f32 {

    fn identity(&self) -> Self {
        *self
    }

    fn odds(&self) -> f32 {
        *self / (1. - *self)
    }

    // The logit or log-odds of a probability p \in [0,1]
    // is the natural log of the ratio p/(1-p)
    fn logit(&self) -> Self {
        let p = if *self == 0.0 {
            f32::EPSILON
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

    fn standardize(&self, mean : &Self, stddev : &Self) -> Self {
        (*self - *mean) / *stddev
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
// for all probs up to the desired point. TODO multiply by -1 outside the summation.
// Perhaps call cumulative_neg_information
pub fn cumulative_entropy<'a>(probs : impl Iterator<Item=f32> + 'a) -> impl Iterator<Item=f32> + 'a {
    running::cumulative_sum(probs.map(move |p| (-1.) * p * p.ln() ))
}

// Calculate the total entropy for an iterator over probabilities.
pub fn entropy(probs : impl Iterator<Item=f32>) -> f32 {
    (-1.)*probs.fold(0.0, |total, p| total + p * p.ln() )
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
    use std::cmp::{PartialEq, Eq, PartialOrd};

    pub struct Accumulated<T> {
        pub pos : usize,
        pub val : T
    }

    // Returns the position of iterated elements up to the point
    // where the cumulative sum equals to or exceeds the val argument
    // (if the iterator ends before that point, return None).
    pub fn cumulative_sum_up_to<T>(iter : impl Iterator<Item=T>, val : T) -> Option<Accumulated<T>>
    where
        T : AddAssign + Zero + Copy + PartialEq + PartialOrd
    {
        let mut cs = cumulative_sum(iter);
        let mut i = 0;
        loop {
            match cs.next() {
                Some(s) => {
                    if s >= val {
                        return Some(Accumulated { pos : i, val : s });
                    }
                },
                None => {
                    return None;
                }
            }
            i += 1;
        }
    }

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
pub mod frequency {

    use super::*;

    pub struct Mode<T> {
        pub pos : usize,
        pub val : T
    }

    pub fn mode<T>(s : &[T])
    where
        T : PartialOrd + Copy
    {
        let (pos, val) = s.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal) )
            .unwrap();
        Mode { pos, val }
    }

}

pub mod rank {

    #[derive(Debug, Clone, Copy)]
    pub struct Rank<T> {
        pub pos : usize,
        pub val : T
    }

    use super::*;

    pub fn quantile_mut<T : Ord + Copy>(s : &mut [T], q : f32) -> Rank<T> {
        assert!(q >= 0.0 && q <= 1.0);
        let pos = (s.len() as f32 * q) as usize;
        let (_, q, _) = s.select_nth_unstable(pos);
        Rank { val : *q, pos }
    }

    pub fn quantile<T : Ord + Copy>(s : &[T], q : f32) -> Rank<T> {
        let mut cs = s.to_vec();
        quantile_mut(&mut cs[..], q)
    }

    pub fn median_mut<T>(s : &mut [T]) -> Rank<T>
    where
        f32 : AsPrimitive<T>,
        T : Add<Output=T> + Mul<Output=T> + Ord + Copy + 'static
    {
        match s.len() % 2 {
            0 => {
                let pos = s.len() / 2;
                let (left, q, _) = s.select_nth_unstable(pos);
                let half : T = (0.5f32).as_();
                let n_left = left.len();
                Rank { val : (left[n_left-1] + *q) * half, pos }
            },
            _ => {
                let pos = s.len() / 2 + 1;
                let (_, q, _) = s.select_nth_unstable(pos);
                Rank { val : *q, pos }
            }
        }
    }

    pub fn median_for_sorted<T>(s : &[T]) -> Rank<T>
    where
        f32 : AsPrimitive<T>,
        T : Add<Output=T> + Mul<Output=T> + Copy + 'static
    {
        match s.len() % 2 {
            0 => {
                let left = s.len() / 2;
                let right = left + 1;
                let half : T = (0.5f32).as_();
                Rank { val : (s[left] + s[right]) * half, pos : left }
            },
            _ => {
                let pos = s.len() / 2 + 1;
                Rank { val : s[pos], pos }
            }
        }
    }

    pub fn median<T>(s : &[T]) -> Rank<T>
    where
        f32 : AsPrimitive<T>,
        T : Add<Output=T> + Mul<Output=T> + Ord + Copy + 'static
    {
        let mut cs = s.to_vec();
        median_mut(&mut cs[..])
    }

    // pub struct Interval<T> { }

    // pub fn range()

    /*pub fn median<T>(v : &[T]) -> T {

    }

    pub fn median_mut(v : &[T]) -> {

    }*/

}

// Calculate sample statistics from contiguous memory regions (slices). Cheaper
// than a running iterator, because there isn't the need to count elements.
pub mod moment {

    /*pub trait Sample
    where
        Self : Add<Output=Self> + Copy + 'static,
        usize : AsPrimitive<Self>
    {

    }*/

    use std::ops::{Add, AddAssign, Div, Sub};
    use num_traits::{Zero, AsPrimitive, Float};

    pub struct FstMoment<T> {
        pub mean : T,
    }

    impl<T> FstMoment<T> {
        pub fn calculate(v : &[T]) -> Self
        where
            T : AddAssign + Copy + Zero +  Div<Output=T> + 'static,
            usize : AsPrimitive<T>
        {
            let mut s = T::zero();
            let n : T = v.len().as_();
            for vi in v {
                s += *vi;
            }
            FstMoment { mean : s / n }
        }

        pub fn calculate_weighted<P>(v : &[T], p : &[T]) -> Self
        where
            T : AddAssign + Copy + Zero + Div<Output=T> + Float + Sub<Output=T> + 'static,
            usize : AsPrimitive<T>
        {
            assert!(v.len() == p.len());
            let mut s = T::zero();
            let n : T = v.len().as_();
            let two = T::from(2.0).unwrap();
            for (vi, pi) in v.iter().zip(p.iter()) {
                s += (*vi) * (*pi);
            }
            FstMoment { mean : s }
        }
    }

    pub struct SndMoment<T> {
        pub mean : T,
        pub var : T
    }

    impl<T> SndMoment<T> {

        pub fn calculate(v : &[T]) -> Self
        where
            T : AddAssign + Copy + Zero + Div<Output=T> + Float + 'static,
            usize : AsPrimitive<T>
        {
            let mut s = T::zero();
            let mut s_sqr = T::zero();
            let n : T = v.len().as_();
            let two = T::from(2.0).unwrap();
            for vi in v {
                s += *vi;
                s_sqr += vi.powf(two);
            }
            let mean = s / n;
            let var = s_sqr / n + mean;
            SndMoment { mean, var }
        }

        pub fn calculate_weighted<P>(v : &[T], p : &[T]) -> Self
        where
            T : AddAssign + Copy + Zero + Div<Output=T> + Float + Sub<Output=T> + 'static,
            usize : AsPrimitive<T>
        {
            assert!(v.len() == p.len());
            let mut s = T::zero();
            let mut s_sqr = T::zero();
            let n : T = v.len().as_();
            let two = T::from(2.0).unwrap();
            for (vi, pi) in v.iter().zip(p.iter()) {
                let w_vi = (*vi) * (*pi);
                s += w_vi;
                s_sqr += w_vi * (*vi);
            }
            SndMoment { mean : s, var : s_sqr - s }
        }
    }

    // The skewness of a normal distribution is 0.0; distributions skewed to the
    // right have a positive skew; distributions skewed to the left have a negative skew.
    pub struct ThrMoment<T> {
        pub mean : T,
        pub var : T,
        pub skew : T
    }

    impl<T> ThrMoment<T> {

        pub fn calculate(v : &[T]) -> Self
        where
            T : AddAssign + Copy + Zero + Div<Output=T> + Float + 'static,
            usize : AsPrimitive<T>
        {
            let mut s = T::zero();
            let mut s_sqr = T::zero();
            let mut s_cub = T::zero();
            let n : T = v.len().as_();
            let two = T::from(2.0).unwrap();
            let three = T::from(3.0).unwrap();
            for vi in v {
                s += *vi;
                s_sqr += vi.powf(two);
                s_cub += vi.powf(three);
            }
            let mean = s / n;
            let var = s_sqr / n + mean;
            let stddev = var.sqrt();
            let skew = (s_cub / n - three*mean*var - mean.powf(three)) / stddev.powf(two);
            ThrMoment { mean, var, skew }
        }
    }

    // The kurtosis of a normal distribution is 3.0, distributions with kurtosis
    // values different than that have more outliers or less outliers than a normal
    // distribution.
    pub struct FthMoment<T> {
        pub mean : T,
        pub var : T,
        pub skew : T,
        pub kurtosis : T
    }

    impl<T> FthMoment<T> {

        pub fn calculate(v : &[T]) -> Self
        where
            T : AddAssign + Copy + Zero + Div<Output=T> + Float + 'static,
            usize : AsPrimitive<T>
        {
            let mut s = T::zero();
            let mut s_sqr = T::zero();
            let mut s_cub = T::zero();
            let mut s_fth = T::zero();
            let n : T = v.len().as_();
            let two = T::from(2.0).unwrap();
            let three = T::from(3.0).unwrap();
            let four = T::from(4.0).unwrap();
            let seven = T::from(7.0).unwrap();
            for vi in v {
                s += *vi;
                s_sqr += vi.powf(two);
                s_cub += vi.powf(three);
                s_fth += vi.powf(four);
            }
            let mean = s / n;
            let var = s_sqr / n + mean;
            let stddev = var.sqrt();
            let skew = (s_cub / n - three*mean*var - mean.powf(three)) / stddev.powf(two);
            let kurtosis = (seven*mean.powf(four) - four*mean*(s_cub / n) * two*mean.powf(two)*var) / var.powf(two);
            FthMoment { mean, var, skew, kurtosis }
        }
    }

}

// Implementor must check if slice is empty.
pub(crate) fn iter_disjoint<T>(s : &[T], ranges : &[Range<usize>], mut f : impl FnMut(&[T])) {
    match ranges.len() {
        0 => {
            f(&s[..]);
        },
        1 => {
            f(&s[0..(ranges[0].start)]);
            f(&s[(ranges[0].end)..(s.len())]);
        },
        n => {
            f(&s[0..(ranges[0].start)]);
            for i in 0..(ranges.len()-1) {
                f(&s[(ranges[i].end)..(ranges[i+1].start)]);
            }
            f(&s[(ranges[n-1].end)..(s.len())]);
        }
    }
}
