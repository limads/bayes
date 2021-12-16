use std;
use std::ops::Add;
use num_traits::Zero;
use num_traits::sign::Unsigned;
use num_traits::cast::{NumCast, ToPrimitive};
use std::ops::AddAssign;
use std::cmp::Ord;

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

    fn unscale(&self, inv_scale : &Self) -> Self {
        *self / *inv_scale
    }

    fn standardize(&self, mean : &Self, inv_scale : &Self) -> Self {
        self.center(mean).unscale(inv_scale)
    }

}

/// Single-pass univariate statistical calculations.
pub mod running {

    use super::*;

    pub fn mean_variance(d : &[f64], unbiased : bool) -> (f64, f64) {
        let n = d.len() as f64;
        let (sum, sum_sq) = d.iter()
            .map(|s| (*s, s.powf(2.)) )
            .fold((0.0, 0.0), |acc, s| (acc.0 + s.0, acc.1 + s.1) );
        let mean = sum / n;
        let var = (sum_sq - sum.powf(2.) / n) / if unbiased { n - 1. } else { n };
        (mean, var)
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

#[test]
fn quantile() {
    let mut s = (0..100).map(|s| 1u64 );
    let s_sum = s.clone().sum::<u64>();
    let q_25 = running::quantile(&mut s, None, s_sum, 0.25).unwrap();
    let q_75 = running::quantile(&mut s, Some(q_25), s_sum, 0.75).unwrap();
    println!("Q25 = {:?}; Q75 = {:?}", q_25, q_75);
    println!("Q25/Q75 = {:?};", running::quantiles((0..100).map(|s| 1u64 ), s_sum, &[0.25, 0.75]));

}
