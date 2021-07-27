use nalgebra::*;
use nalgebra::storage::*;
use std::cmp::Ordering;
use super::*;
use std::iter::FromIterator;
use std::fmt;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

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
    bins : HashMap<u64, u64>
}

#[derive(Serialize, Deserialize)]
pub struct Bin {
    pub low : f64,
    pub high : f64,
    pub count : u64,
    pub prop : f64
}

impl Histogram {

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
        let mut bins = HashMap::<u64, u64>::new();
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
