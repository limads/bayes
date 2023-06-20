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
use num_traits::Float;

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
        1. / ((1. + (-1.0 * (*self)).exp() ) + std::f64::EPSILON)
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
        1. / ((1. + (-1. * (*self)).exp() ) + std::f32::EPSILON)
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

pub fn accumulate<T>(mut v : Vec<T>) -> Vec<T>
where
    T : AddAssign + Copy
{
    accumulate_inplace(&mut v);
    v
}

pub fn accumulate_inplace<T>(vals : &mut impl AsMut<[T]>)
where
    T : AddAssign + Copy
{
    let mut vals = vals.as_mut();
    for i in 1..vals.len() {
        vals[i] += vals[i-1];
    }
}

pub fn histogram_mean_stddev<T, U>(domain : &[T], vals : &[U]) -> (T, T)
where
    T : num_traits::AsPrimitive<f32>,
    U : num_traits::AsPrimitive<f32>,
    f32 : num_traits::AsPrimitive<T>
{
    assert!(domain.len() == vals.len());
    let mut sum : f32 = 0.0;
    let mut sum_sq : f32 = 0.0;
    let mut n = 0.0;
    for i in 0..domain.len() {
        let d : f32 = domain[i].as_();
        let v : f32 = vals[i].as_();
        sum += d*v;
        sum_sq += d.powf(2.)*v;
        n += v;
    }
    let mean = sum / n;
    (mean.as_(), ((sum_sq / n) - mean.powf(2.)).as_())
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
pub fn cumulative_entropy<'a, F>(probs : impl Iterator<Item=F> + 'a) -> impl Iterator<Item=F> + 'a
where
    F : Float + Mul + Add + From<f32> + AddAssign + 'static
{
    running::cumulative_sum(probs.map(move |p| <F as From<f32>>::from((-1.)) * (p + F::epsilon()) * (p + F::epsilon()).ln() ))
}

// Calculate the total entropy for an iterator over probabilities.
pub fn entropy<F>(probs : impl Iterator<Item=F>) -> F
where
    F : Float + Mul + Add + From<f32> + Zero
{
    <F as From<f32>>::from(-1.) * probs.fold(F::zero(), |total, p| total + (p + F::epsilon()) * (p + F::epsilon()).ln() )
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
    
    /* Anything calculated from a cumulative sum can have intermediate values
    calculated in O(1) by indexing the difference over a range. But The first range
    is a special case, since the cumulative sum is simply the regular value
    (cumulative sums start at first element, not at zero). */
    pub fn accumulated_range<T>(accum : &[T], range : Range<usize>) -> T 
    where
        T : Sub<Output=T> + Copy
    {
        if range.start > 0 {
            accum[range.end-1] - accum[range.start]
        } else {
            accum[range.end-1]
        }
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

// TODO rename to RankSummary. Then create struct Summary { ranks : moments : }
// That contain the rank summary and n-th (4) moment summary.
pub struct Ranks<T> {
    pub min : T,
    pub max : T,
    pub q25 : T,
    pub q75 : T,
    pub median : T
}

impl<T> Ranks<T> {

    // pub fn interquartile_range(&self) -> T {
    //    self.q75 - self.q25
    // }

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

// Count-based descriptive statistics (mode)
// TODO calculate mean with slice::select_nth_unstable_by
// or select_nth_unstable. Perhaps rename to freq?
pub mod count {

    use super::*;

    #[derive(Debug, Clone)]
    pub struct Mode<T>
    where T : Copy
    {
        pub pos : usize,
        pub val : T
    }

    impl<T> PartialEq for Mode<T>
    where
        T : PartialEq + Copy
    {
        fn eq(&self, other : &Self) -> bool {
            self.val.eq(&other.val)
        }
    }

    impl<T> PartialOrd for Mode<T>
    where
        T : PartialOrd + Copy
    {
        fn partial_cmp(&self, other : &Self) -> Option<Ordering> {
            self.val.partial_cmp(&other.val)
        }
    }

    pub fn mode<T>(s : &[T]) -> Mode<T>
    where
        T : PartialOrd + Copy
    {
        let (pos, val) = s.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal) )
            .unwrap();
        Mode { pos, val : *val }
    }

}

/// Rank-based descriptive statistics (median, quantiles)
pub mod rank {

    use super::*;
    use smallvec::SmallVec;

    #[derive(Debug, Clone)]
    pub struct OnlineMedian<T : PartialOrd + Copy> {
        data : Vec<T>,
        ranks : Vec<usize>,
        order : usize
    }

    impl<T : PartialOrd + Copy> OnlineMedian<T> {

        pub fn data(&self) -> &[T] {
            &self.data[..]
        }

        pub fn len(&self) -> usize {
            self.data.len()
        }

        pub fn clear(&mut self) {
            self.data.clear();
        }

        pub fn new(order : usize) -> Self {
            assert!(order % 2 != 0);
            Self { order, data : Vec::with_capacity(order), ranks : Vec::with_capacity(order) }
        }

        pub fn state(&self) -> Option<T> {
            self.data.get(self.data.len() / 2).cloned()
        }

        pub fn batch_update(&mut self, dts : &[T]) -> Vec<T> {
            dts.iter().filter_map(|dt| self.update(*dt) ).collect()
        }

        pub fn update(&mut self, dt : T) -> Option<T> {
            if self.data.len() < self.order {
                match self.data.binary_search_by(|d| d.partial_cmp(&dt).unwrap_or(Ordering::Equal) ) {
                    Ok(ix) | Err(ix) => {
                        self.data.insert(ix, dt);
                        self.ranks.push(ix);
                    }
                }
            } else {
                let oldest = self.ranks.remove(0);
                self.data.remove(oldest);
                match self.data.binary_search_by(|d| d.partial_cmp(&dt).unwrap_or(Ordering::Equal) ) {
                    Ok(ix) | Err(ix) => {
                        self.data.insert(ix, dt);
                        self.ranks.push(ix);
                    }
                }
            }
            self.state()
        }

    }

    #[derive(Debug, Clone)]
    pub struct Extrema<T> {
        pub min : T,
        pub max : T
    }

    impl<T> Extrema<T>
    where
        T : PartialOrd + num_traits::Bounded + Copy
    {

        pub fn batch_update(&mut self, vals : &[T]) {
            for val in vals {
                self.update(*val);
            }
        }

        pub fn calculate(vals : &[T]) -> Self {
            let mut extr = Extrema::new();
            extr.batch_update(vals);
            extr
        }

        pub fn new() -> Self {
            Self { min : T::max_value(), max : T::min_value() }
        }

        pub fn update(&mut self, val : T) {
            if val > self.max {
                self.max = val;
            }
            if val < self.min {
                self.min = val;
            }
        }

    }

    #[derive(Debug, Clone, Copy)]
    pub struct Rank<T> {
        pub pos : usize,
        pub val : T
    }

    pub struct Ranked<T>
    where
        T : PartialOrd
    {
        vals : SmallVec<[T; 8]>,
        curr : usize
    }

    pub struct RankedBy<T, F>
    where
        T : PartialOrd,
        F : Fn(&T, &T)->Ordering
    {
        vals : SmallVec<[T; 8]>,
        f : F,
        curr : usize
    }

    impl<T> Ranked<T>
    where
        T : PartialOrd + Copy
    {

        fn update(&mut self, it : T) -> Rank<T> {
            match self.vals.binary_search_by(|probe| probe.partial_cmp(&it).unwrap_or(Ordering::Equal) ) {
                Ok(pos) => {
                    self.vals.insert(pos+1, it);
                    Rank { val : it, pos : pos + 1}
                },
                Err(pos) => {
                    self.vals.insert(pos, it);
                    Rank { val : it, pos : pos }
                }
            }
        }

        // pub fn sorted_values(self) -> SmallVec<T> {
        //     self.vals
        // }

    }

    impl<T, F> RankedBy<T, F>
    where
        T : PartialOrd + Copy,
        F : Fn(&T, &T)->Ordering
    {
        fn update(&mut self, it : T) -> Rank<T> {
            match self.vals.binary_search_by(|probe| (&self.f)(probe, &it) ) {
                Ok(pos) => {
                    self.vals.insert(pos+1, it);
                    Rank { val : it, pos : pos + 1}
                },
                Err(pos) => {
                    self.vals.insert(pos, it);
                    Rank { val : it, pos : pos }
                }
            }
        }
    }

    pub trait Ranks<T>
    where
        Self : Sized,
        Self : Iterator<Item=T>,
        T : PartialOrd
    {

        /* Iterator adaptor that return the rank of the ith item. */
        fn rank(self) -> Ranked<T>;

        fn rank_by<F>(self, f : F) -> RankedBy<T, F>
        where
            F : Fn(&T, &T)->Ordering;

    }

    impl<I, T> Ranks<T> for I
    where
        I : Sized + Iterator<Item=T>,
        T : PartialOrd
    {

        fn rank(self) -> Ranked<T> {
            let mut vals : SmallVec<[T; 8]> = self.collect();
            vals.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal) );
            Ranked { vals, curr : 0 }
        }

        fn rank_by<F>(self, f : F) -> RankedBy<T, F>
        where
            F : Fn(&T, &T)->Ordering
        {
            let mut vals : SmallVec<[T; 8]> = self.collect();
            vals.sort_by(&f);
            RankedBy { vals, curr : 0, f }
        }

    }

    impl<T> Iterator for Ranked<T>
    where
        T : PartialOrd + Copy
    {

        type Item = Rank<T>;

        fn next(&mut self) -> Option<Self::Item> {
            let old_curr = self.curr;
            let it = self.vals.get(old_curr)?;
            self.curr += 1;
            Some(Rank { pos : old_curr, val : *it })
        }

    }

    impl<T, F> Iterator for RankedBy<T, F>
    where
        T : PartialOrd + Copy,
        F : Fn(&T, &T)->Ordering + Clone
    {

        type Item = Rank<T>;

        fn next(&mut self) -> Option<Self::Item> {
            let old_curr = self.curr;
            let it = self.vals.get(old_curr)?;
            self.curr += 1;
            Some(Rank { pos : old_curr, val : *it })
        }

    }

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

    // Premultiply both by 0.5 and do the cast to avoid integer overflows
    // (specially for u8 and u16).
    fn even_median<T>(left : &[T], q : T) -> T
    where
        f32 : AsPrimitive<T>,
        T : AsPrimitive<f32>
   {
        let n_left = left.len();
        let a : f32 = left[n_left-1].as_();
        let b : f32 = q.as_();
        (0.5*a + 0.5*b).as_()
    }

    pub fn median_mut<T>(s : &mut [T]) -> Option<Rank<T>>
    where
        f32 : AsPrimitive<T>,
        T : Add<Output=T> + Mul<Output=T> + Ord + Copy + 'static + AsPrimitive<f32>
    {
        match s.len() % 2 {
            0 => {
                let pos = s.len() / 2;
                if pos >= s.len() {
                    return None;
                }
                let (left, q, _) = s.select_nth_unstable(pos);
                Some(Rank { val : even_median(left, *q), pos })
            },
            _ => {
                let pos = s.len() / 2 + 1;
                if pos >= s.len() {
                    return None;
                }
                let (_, q, _) = s.select_nth_unstable(pos);
                Some(Rank { val : *q, pos })
            }
        }
    }

    pub fn median_for_sorted<T>(s : &[T]) -> Option<Rank<T>>
    where
        f32 : AsPrimitive<T>,
        T : Add<Output=T> + Mul<Output=T> + Copy + 'static + AsPrimitive<f32>
    {
        if s.len() == 0 {
            return None;
        }
        match s.len() % 2 {
            0 => {
                let left = s.len() / 2;
                let right = left + 1;
                Some(Rank { val : even_median(&s[..=left], s[right]), pos : left })
            },
            _ => {
                let pos = s.len() / 2 + 1;
                Some(Rank { val : s[pos], pos })
            }
        }
    }

    pub fn median_for_iter<I, T>(iter : I) -> Option<Rank<T>>
    where
        I : Iterator<Item=T>,
        f32 : AsPrimitive<T>,
        T : Add<Output=T> + Mul<Output=T> + Ord + Copy + 'static + AsPrimitive<f32>
    {
        let mut v : Vec<T> = iter.collect();
        median_mut(&mut v[..])
    }

    pub fn median<T>(s : &[T]) -> Option<Rank<T>>
    where
        f32 : AsPrimitive<T>,
        T : Add<Output=T> + Mul<Output=T> + Ord + Copy + 'static + AsPrimitive<f32>
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

    /* Keeps a sorted collection of objects with a size upper bound.
    As elements are added, old elements close to the rank of the
    added element are removed to keep the size constant. */
    #[derive(Debug, Clone)]
    pub struct RankSet<T> {
        vals : Vec<T>,
        sz : usize
    }

    impl<T> RankSet<T>
    where
        T : num_traits::Bounded + Clone + Copy + PartialOrd,
        T : num_traits::AsPrimitive<f32>
    {

        pub fn values(&self) -> &[T] {
            &self.vals[..]
        }

        pub fn split_quantile(&self, q : f64) -> (&[T], &[T]) {
            let pos = (self.vals.len() as f64 * q) as usize;
            self.vals.split_at(pos)
        }

        pub fn quantile(&self, q : f64) -> Option<Rank<T>> {
            let pos = (self.vals.len() as f64 * q) as usize;
            Some(Rank { val : self.vals.get(pos)?.clone(), pos })
        }

        pub fn extrema(&self) -> Option<Extrema<T>> {
            Some(Extrema { min : self.vals.first()?.clone(), max : self.vals.last()?.clone() })
        }

        pub fn rank(&self, rank : usize) -> Option<Rank<T>> {
            Some(Rank { val : self.vals.get(rank)?.clone(), pos : rank })
        }

        pub fn new(sz : usize) -> Self {
            assert!(sz >= 2);
            Self {
                vals : Vec::new(),
                sz
            }
        }

        pub fn add(&mut self, val : T) {
            if self.vals.len() == 0 {
                self.vals.push(val);
                return;
            }
            let last = self.vals[self.vals.len()-1];
            let first = self.vals[0];
            if val > last {
                self.vals.push(val);
            } else if val < first {
                self.vals.insert(0, val);
            } else if self.vals.len() < self.sz {
                match self.vals.binary_search_by(|old| old.partial_cmp(&val).unwrap_or(Ordering::Equal) ) {
                    Ok(_) => { },
                    Err(pos) => {
                        self.vals.insert(pos, val);
                    }
                }
            } else {
                match self.vals.binary_search_by(|old| old.partial_cmp(&val).unwrap_or(Ordering::Equal) ) {
                    Ok(_) => { },
                    Err(pos) => {
                        if pos == 0 || pos == self.vals.len() - 1 {
                            self.vals[pos] = val;
                        } else {
                            let diff_left : f32 = (self.vals[pos].as_() - val.as_()).abs();
                            let diff_right : f32 = (self.vals[pos+1].as_() - val.as_()).abs();
                            if diff_right > diff_left {
                                self.vals[pos+1] = val;
                            } else {
                                self.vals[pos] = val;
                            }
                        }
                    }
                }
            }
        }

    }

}

pub fn geometric_mean(xs : &[f64]) -> f64 {
    let avg_ln = xs.iter().fold(0.0, |m, x| m + x.ln() ) / xs.len() as f64;
    avg_ln.exp()
}

pub fn harmonic_mean(xs : &[f64]) -> f64 {
    let avg_inv = xs.iter().fold(0.0, |m, x| m + 1.0 / x ) / xs.len() as f64;
    1.0 / avg_inv
}

pub trait Mean<T> {

    fn mean(self) -> T;

    // geometric_mean()

    // harmonic_mean()

    // power_mean() generalized mean

    fn casting_mean(self) -> T;

}

use std::ops::Div;

impl<I, T> Mean<T> for I
where
    I : Iterator<Item=T>,
    T : Add<Output=T> + Div<Output=T> + num_traits::Zero + Copy + num_traits::ToPrimitive + 'static,
    usize : num_traits::AsPrimitive<T>,
    f32 : num_traits::AsPrimitive<T>,

{

    fn casting_mean(self) -> T {
        let mut n : f32 = 0.;
        let s : f32 = self.fold(T::zero().to_f32().unwrap(), |s, t| { n += 1.; s + t.to_f32().unwrap() });
        (s / n).as_()
    }

    fn mean(self) -> T {
        let mut n = 0;
        let s : T = self.fold(T::zero(), |s, t| { n += 1; s + t });
        let nt : T = n.as_();
        s / nt
    }

}

// Moment-based descriptive statistics (mean, variance, skew, kurtosis).
// Calculate sample statistics from contiguous memory regions (slices). Cheaper
// than a running iterator, because there isn't the need to count elements.
pub mod moment {

    /*
    A moment is the expected value of the power of a random variable.
    Several statistical quantities of interest can be expressed as moments
    of random variables or of simple functions of random variables. Knowledge
    about the moments of a random variable restrict the range of values the
    random variable can take. This module contains functions for the estimation
    of sample random moments.
    */

    /*pub trait Sample
    where
        Self : Add<Output=Self> + Copy + 'static,
        usize : AsPrimitive<Self>
    {

    }*/

    use std::ops::{Add, AddAssign, Div, Sub};
    use num_traits::{Zero, AsPrimitive, Float, Pow};

    pub struct Accumulator<N>
    where
        N : Copy + Add + AddAssign + Zero + Pow<N, Output=N> + 'static,
        N : AsPrimitive<f32>,
        usize : AsPrimitive<N>,
        f32 : AsPrimitive<N>
    {
        v : N,
        v2 : N,
        v3 : N,
        v4 : N,
        n : usize
    }

    impl<N> Accumulator<N>
    where
        N : Copy + Add + AddAssign + Zero + Pow<N, Output=N> + 'static,
        N : AsPrimitive<f32>,
        usize : AsPrimitive<N>,
        f32 : AsPrimitive<N>
    {

        pub fn new(n_moments : usize) -> Self {
            Self { v : N::zero(), v2 : N::zero(), v3 : N::zero(), v4 : N::zero(), n : 0 }
        }

        pub fn increment(&mut self, val : N) {
            self.n += 1;
            self.v += val;
            self.v2 += val.pow((2usize).as_());
        }

        pub fn second_moment(&self) -> SndMoment<N> {
            let sum : f32 = self.v.as_();
            let sum_sq : f32 = self.v2.as_();
            let n : f32 = self.n as f32;
            let mean : N = (sum / n).as_();
            let var : N = ((sum_sq - sum.powf(2.) / n) / n).as_();
            SndMoment { mean, var }
        }

    }

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

pub mod var {

    pub fn variance() { }

    pub fn stddev() { }

    pub fn absdev() { }

    // Iterator adaptor. Then create another iterator adaptor that
    // returns standardized values.
    // pub trait Dispersion where Self : Mean { fn variance() fn stddev() absdev() }
    // pub trait Standardized { }

}

pub mod corr {

    pub fn pearson() { }

    pub fn spearman() { }

    pub fn kendall() { }

}

#[test]
fn online_median() {
    let mut om = crate::calc::rank::OnlineMedian::<f32>::new(5);
    for it in [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0] {
        println!("{:?} {:?} {:?}", om.update(it), om.len(), om.data());
    }
}

